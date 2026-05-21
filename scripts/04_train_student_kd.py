# -*- coding: utf-8 -*-
"""
04_train_student_kd.py

Train student models with different KD methods.

Methods:
    student_only
    single_kd
    average_kd
    confidence_kd
    dream_kd

Examples:
    python scripts/04_train_student_kd.py --method student_only
    python scripts/04_train_student_kd.py --method average_kd
    python scripts/04_train_student_kd.py --method confidence_kd
    python scripts/04_train_student_kd.py --method dream_kd
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")
DATA_DIR = PROJECT_DIR / "data_processed"
LOGIT_DIR = PROJECT_DIR / "outputs" / "teacher_logits"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "students"

TEACHERS = ["bert", "roberta", "electra"]
SPLITS = ["train", "dev", "test_in", "test_ood"]

NUM_LABELS = 2


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StudentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: np.ndarray, teacher_logits: np.ndarray, tokenizer, max_length: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = labels.astype(np.int64)
        self.teacher_logits = teacher_logits.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {k: torch.tensor(v) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["teacher_logits"] = torch.tensor(self.teacher_logits[idx], dtype=torch.float)
        return item


class DREAMKDHead(nn.Module):
    """
    DeepSets-style disagreement encoder.

    Input:
        teacher_probs: [B, M, C]

    Output:
        teacher_weights: [B, M]
        kd_strength: [B, 1]
    """
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_teachers = num_teachers
        self.num_classes = num_classes

        self.phi = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.teacher_score = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.kd_strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, teacher_probs):
        # teacher_probs: [B, M, C]
        B, M, C = teacher_probs.shape

        h = self.phi(teacher_probs)        # [B, M, H]
        pooled = h.mean(dim=1)             # [B, H], permutation-invariant
        z = self.rho(pooled)               # [B, H]

        z_expand = z.unsqueeze(1).expand(-1, M, -1)     # [B, M, H]
        score_input = torch.cat([z_expand, teacher_probs], dim=-1)
        teacher_scores = self.teacher_score(score_input).squeeze(-1)  # [B, M]
        teacher_weights = F.softmax(teacher_scores, dim=-1)

        kd_strength = self.kd_strength_head(z)  # [B, 1]

        return teacher_weights, kd_strength


class MLPTeacherGating(nn.Module):
    """
    Stronger baseline: concatenate teacher probabilities and produce teacher weights.
    This is not permutation-invariant, but it is a learned teacher weighting baseline.
    """
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_teachers = num_teachers
        self.num_classes = num_classes
        in_dim = num_teachers * num_classes

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_teachers),
        )

    def forward(self, teacher_probs):
        B, M, C = teacher_probs.shape
        x = teacher_probs.reshape(B, M * C)
        scores = self.net(x)
        weights = F.softmax(scores, dim=-1)
        return weights


def load_split(split_name: str):
    df = pd.read_csv(DATA_DIR / f"{split_name}.csv")
    labels = np.load(LOGIT_DIR / f"{split_name}_labels.npy")

    logits_list = []
    for teacher in TEACHERS:
        logits = np.load(LOGIT_DIR / f"{teacher}_{split_name}_logits.npy")
        logits_list.append(logits)

    # [N, M, C]
    teacher_logits = np.stack(logits_list, axis=1)
    return df, labels, teacher_logits


def build_dataloaders(tokenizer, max_length: int, batch_size: int):
    data = {}
    for split in SPLITS:
        df, labels, teacher_logits = load_split(split)
        dataset = StudentDataset(df, labels, teacher_logits, tokenizer, max_length)
        shuffle = split == "train"
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        data[split] = loader
    return data


def compute_metrics_from_logits(logits, labels):
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
    }


def kd_kl_loss(student_logits, teacher_probs, temperature: float):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return loss


def get_teacher_target(args, teacher_logits, dream_head=None, mlp_gate=None):
    """
    teacher_logits: [B, M, C]
    Returns:
        fused_teacher_probs: [B, C]
        kd_strength: [B, 1]
        aux: dict
    """
    temperature = args.temperature
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # [B, M, C]

    aux = {}

    if args.method == "single_kd":
        # use the strongest teacher: electra, index 2
        fused = teacher_probs[:, 2, :]
        kd_strength = torch.ones((teacher_probs.size(0), 1), device=teacher_probs.device)
        return fused, kd_strength, aux

    if args.method == "average_kd":
        fused = teacher_probs.mean(dim=1)
        kd_strength = torch.ones((teacher_probs.size(0), 1), device=teacher_probs.device)
        return fused, kd_strength, aux

    if args.method == "confidence_kd":
        # weight by max confidence of each teacher
        conf = teacher_probs.max(dim=-1).values  # [B, M]
        weights = F.softmax(conf, dim=-1)
        fused = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)
        kd_strength = torch.ones((teacher_probs.size(0), 1), device=teacher_probs.device)

        aux["teacher_weights"] = weights.detach()
        return fused, kd_strength, aux

    if args.method == "mlp_gating_kd":
        weights = mlp_gate(teacher_probs)
        fused = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)
        kd_strength = torch.ones((teacher_probs.size(0), 1), device=teacher_probs.device)

        aux["teacher_weights"] = weights.detach()
        return fused, kd_strength, aux

    if args.method == "dream_kd":
        weights, kd_strength = dream_head(teacher_probs)
        fused = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)

        # Avoid completely turning off KD in early experiments.
        kd_strength = args.lambda_min + (args.lambda_max - args.lambda_min) * kd_strength

        aux["teacher_weights"] = weights.detach()
        aux["kd_strength"] = kd_strength.detach()
        return fused, kd_strength, aux

    raise ValueError(f"Unsupported KD method: {args.method}")


def evaluate(model, data_loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ["labels", "teacher_logits"]
            }

            outputs = model(**inputs)
            logits = outputs.logits

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return compute_metrics_from_logits(all_logits, all_labels)


def train(args):
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Training student")
    print("=" * 80)
    print(f"Method:      {args.method}")
    print(f"Student:     {args.student_model}")
    print(f"Device:      {device}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    run_dir = OUTPUT_DIR / args.method
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=NUM_LABELS,
    )
    model.to(device)

    loaders = build_dataloaders(tokenizer, args.max_length, args.batch_size)

    dream_head = None
    mlp_gate = None

    extra_modules = []

    if args.method == "dream_kd":
        dream_head = DREAMKDHead(
            num_teachers=len(TEACHERS),
            num_classes=NUM_LABELS,
            hidden_dim=args.gate_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        extra_modules.append(dream_head)

    if args.method == "mlp_gating_kd":
        mlp_gate = MLPTeacherGating(
            num_teachers=len(TEACHERS),
            num_classes=NUM_LABELS,
            hidden_dim=args.gate_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        extra_modules.append(mlp_gate)

    params = list(model.parameters())
    for module in extra_modules:
        params += list(module.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_training_steps = len(loaders["train"]) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_dev_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in extra_modules:
            module.train()

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0

        progress = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress:
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)

            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ["labels", "teacher_logits"]
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**inputs)
                student_logits = outputs.logits

                ce_loss = F.cross_entropy(student_logits, labels)

                if args.method == "student_only":
                    loss = ce_loss
                    kd_loss_value = torch.tensor(0.0, device=device)
                else:
                    teacher_target, kd_strength, aux = get_teacher_target(
                        args=args,
                        teacher_logits=teacher_logits,
                        dream_head=dream_head,
                        mlp_gate=mlp_gate,
                    )

                    # sample-wise KL
                    student_log_probs = F.log_softmax(student_logits / args.temperature, dim=-1)
                    per_sample_kd = F.kl_div(
                        student_log_probs,
                        teacher_target,
                        reduction="none",
                    ).sum(dim=-1) * (args.temperature ** 2)

                    kd_loss_value = (kd_strength.squeeze(-1) * per_sample_kd).mean()

                    loss = args.ce_weight * ce_loss + args.kd_weight * kd_loss_value

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kd += kd_loss_value.item()

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}",
                "kd": f"{kd_loss_value.item():.4f}",
            })

        dev_metrics = evaluate(model, loaders["dev"], device)
        test_in_metrics = evaluate(model, loaders["test_in"], device)
        test_ood_metrics = evaluate(model, loaders["test_ood"], device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / len(loaders["train"]),
            "train_ce": total_ce / len(loaders["train"]),
            "train_kd": total_kd / len(loaders["train"]),
            "dev": dev_metrics,
            "test_in": test_in_metrics,
            "test_ood": test_ood_metrics,
        }
        history.append(epoch_record)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} results")
        print("=" * 80)
        print(f"Dev macro-F1:      {dev_metrics['macro_f1']:.4f}")
        print(f"Test-in macro-F1:  {test_in_metrics['macro_f1']:.4f}")
        print(f"Test-OOD macro-F1: {test_ood_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]

            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "dev_macro_f1": best_dev_f1,
            }

            if dream_head is not None:
                best_state["dream_head"] = {k: v.cpu() for k, v in dream_head.state_dict().items()}

            if mlp_gate is not None:
                best_state["mlp_gate"] = {k: v.cpu() for k, v in mlp_gate.state_dict().items()}

    # save history
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        model.to(device)

        if dream_head is not None and "dream_head" in best_state:
            dream_head.load_state_dict(best_state["dream_head"])
            dream_head.to(device)

        if mlp_gate is not None and "mlp_gate" in best_state:
            mlp_gate.load_state_dict(best_state["mlp_gate"])
            mlp_gate.to(device)

    final_dev = evaluate(model, loaders["dev"], device)
    final_test_in = evaluate(model, loaders["test_in"], device)
    final_test_ood = evaluate(model, loaders["test_ood"], device)

    final_results = {
        "method": args.method,
        "student_model": args.student_model,
        "best_dev_macro_f1": best_dev_f1,
        "dev": final_dev,
        "test_in": final_test_in,
        "test_ood": final_test_ood,
        "args": vars(args),
    }

    with open(run_dir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    model.save_pretrained(run_dir / "best_model")
    tokenizer.save_pretrained(run_dir / "best_model")

    if dream_head is not None:
        torch.save(dream_head.state_dict(), run_dir / "dream_head.pt")

    if mlp_gate is not None:
        torch.save(mlp_gate.state_dict(), run_dir / "mlp_gate.pt")

    print("\n" + "=" * 80)
    print("Final best results")
    print("=" * 80)
    print(json.dumps(final_results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "student_only",
            "single_kd",
            "average_kd",
            "confidence_kd",
            "mlp_gating_kd",
            "dream_kd",
        ],
    )

    parser.add_argument("--student_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--kd_weight", type=float, default=0.5)

    # DREAM-KD: restrict lambda to avoid collapse
    parser.add_argument("--lambda_min", type=float, default=0.2)
    parser.add_argument("--lambda_max", type=float, default=1.0)

    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()