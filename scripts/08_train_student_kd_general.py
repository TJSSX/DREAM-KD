# -*- coding: utf-8 -*-
"""
08_train_student_kd_general.py

Generic student training script for multi-teacher KD.

Supported methods:
    student_only
    single_kd
    average_kd
    confidence_kd
    entropy_kd
    mlp_gating_kd
    dream_kd

Key update:
    - student_only does NOT require teacher logits.
    - KD methods require teacher logits.

Example: student-only on DBpedia
    python scripts/08_train_student_kd_general.py ^
        --dataset_name dbpedia ^
        --data_dir data_processed/dbpedia ^
        --teachers none ^
        --method student_only ^
        --student_model google/bert_uncased_L-4_H-256_A-4 ^
        --epochs 1 ^
        --batch_size 32 ^
        --fp16

Example: DREAM-KD on DBpedia after exporting teacher logits
    python scripts/08_train_student_kd_general.py ^
        --dataset_name dbpedia ^
        --data_dir data_processed/dbpedia ^
        --teachers bert_e1 roberta_e1 electra_e1 xlnet_e1 ^
        --method dream_kd ^
        --student_model google/bert_uncased_L-4_H-256_A-4 ^
        --epochs 3 ^
        --batch_size 32 ^
        --fp16
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


PROJECT_DIR = Path(r"D:\Pycharm_Work\DREAM_KD")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_num_labels(data_dir: Path, train_df: pd.DataFrame) -> int:
    info_path = data_dir / "dataset_info.json"

    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if "num_labels" in info:
            return int(info["num_labels"])

    return int(train_df["label"].nunique())


class StudentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        teacher_logits,
        tokenizer,
        max_length: int,
    ):
        self.texts = df["text"].astype(str).tolist()
        self.labels = labels.astype(np.int64)

        # For student_only, teacher_logits is None.
        # For KD methods, teacher_logits shape is [N, M, C].
        self.teacher_logits = teacher_logits

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

        if self.teacher_logits is not None:
            item["teacher_logits"] = torch.tensor(
                self.teacher_logits[idx],
                dtype=torch.float,
            )

        return item


class MLPTeacherGating(nn.Module):
    """
    Learned teacher-weighting baseline.

    Input:
        teacher_probs: [B, M, C]

    Output:
        teacher_weights: [B, M]
    """

    def __init__(
        self,
        num_teachers: int,
        num_labels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        in_dim = num_teachers * num_labels

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_teachers),
        )

    def forward(self, teacher_probs):
        bsz, num_teachers, num_labels = teacher_probs.shape
        x = teacher_probs.reshape(bsz, num_teachers * num_labels)
        scores = self.net(x)
        weights = F.softmax(scores, dim=-1)
        return weights


class DREAMKDHead(nn.Module):
    """
    DREAM-KD v2:
    Disagreement-profile-enhanced DeepSets encoder.

    For each teacher, we construct:
        [teacher_probs, confidence, entropy, L1 deviation, KL deviation]

    Input:
        teacher_probs: [B, M, C]

    Output:
        teacher_weights: [B, M]
        kd_strength: [B, 1]
    """

    def __init__(
        self,
        num_teachers: int,
        num_labels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_teachers = num_teachers
        self.num_labels = num_labels

        # Per-teacher feature:
        # probs C + confidence 1 + entropy 1 + L1 deviation 1 + KL deviation 1
        self.teacher_feature_dim = num_labels + 4

        self.phi = nn.Sequential(
            nn.Linear(self.teacher_feature_dim, hidden_dim),
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
            nn.Linear(hidden_dim + self.teacher_feature_dim, hidden_dim),
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

    @staticmethod
    def entropy(probs, eps=1e-8):
        return -(probs * torch.log(probs + eps)).sum(dim=-1, keepdim=True)

    def build_teacher_features(self, teacher_probs):
        """
        teacher_probs: [B, M, C]

        return:
            teacher_features: [B, M, C + 4]
        """
        eps = 1e-8

        confidence = teacher_probs.max(dim=-1, keepdim=True).values  # [B, M, 1]
        entropy = self.entropy(teacher_probs, eps=eps)               # [B, M, 1]

        mean_probs = teacher_probs.mean(dim=1, keepdim=True)         # [B, 1, C]

        l1_dev = torch.abs(teacher_probs - mean_probs).mean(
            dim=-1,
            keepdim=True,
        )

        kl_dev = (
            teacher_probs
            * (torch.log(teacher_probs + eps) - torch.log(mean_probs + eps))
        ).sum(dim=-1, keepdim=True)

        features = torch.cat(
            [teacher_probs, confidence, entropy, l1_dev, kl_dev],
            dim=-1,
        )

        return features

    def forward(self, teacher_probs):
        # teacher_probs: [B, M, C]
        teacher_features = self.build_teacher_features(teacher_probs)  # [B, M, C+4]

        h = self.phi(teacher_features)  # [B, M, H]
        pooled = h.mean(dim=1)          # [B, H], permutation-invariant
        z = self.rho(pooled)            # [B, H]

        z_expand = z.unsqueeze(1).expand(-1, teacher_probs.size(1), -1)
        score_input = torch.cat([z_expand, teacher_features], dim=-1)

        teacher_scores = self.teacher_score(score_input).squeeze(-1)  # [B, M]
        teacher_weights = F.softmax(teacher_scores, dim=-1)

        kd_strength = self.kd_strength_head(z)  # [B, 1]

        return teacher_weights, kd_strength


def compute_metrics_from_logits(logits, labels):
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
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


def load_split(
    data_dir: Path,
    logit_dir: Path,
    split_name: str,
    teachers: list[str],
    method: str,
):
    """
    For student_only:
        read labels directly from csv, do not require teacher logits.

    For KD methods:
        read teacher logits from outputs/<dataset>/teacher_logits.
    """
    df = pd.read_csv(data_dir / f"{split_name}.csv")

    if method == "student_only":
        labels = df["label"].astype(int).values
        teacher_logits = None
        return df, labels, teacher_logits

    labels_path = logit_dir / f"{split_name}_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Cannot find labels: {labels_path}")

    labels = np.load(labels_path)

    logits_list = []

    for teacher in teachers:
        if teacher.lower() == "none":
            continue

        logits_path = logit_dir / f"{teacher}_{split_name}_logits.npy"

        if not logits_path.exists():
            raise FileNotFoundError(f"Cannot find logits: {logits_path}")

        logits = np.load(logits_path)
        logits_list.append(logits)

    if len(logits_list) == 0:
        raise ValueError(
            "No teacher logits loaded. For KD methods, please provide valid teacher names."
        )

    teacher_logits = np.stack(logits_list, axis=1)  # [N, M, C]
    return df, labels, teacher_logits


def build_dataloaders(
    data_dir: Path,
    logit_dir: Path,
    teachers: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    method: str,
):
    loaders = {}

    for split in ["train", "dev", "test"]:
        df, labels, teacher_logits = load_split(
            data_dir=data_dir,
            logit_dir=logit_dir,
            split_name=split,
            teachers=teachers,
            method=method,
        )

        dataset = StudentDataset(
            df=df,
            labels=labels,
            teacher_logits=teacher_logits,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )

        if teacher_logits is None:
            print(f"{split}: {len(dataset)} examples | teacher_logits=None")
        else:
            print(f"{split}: {len(dataset)} examples | teacher_logits={teacher_logits.shape}")

    return loaders


def get_teacher_target(args, teacher_logits, dream_head=None, mlp_gate=None):
    """
    teacher_logits: [B, M, C]

    Returns:
        teacher_target: [B, C]
        kd_strength: [B, 1]
        aux: dict
    """
    teacher_probs = F.softmax(teacher_logits / args.temperature, dim=-1)
    bsz, num_teachers, num_labels = teacher_probs.shape

    aux = {}

    if args.method == "single_kd":
        index = min(args.single_teacher_index, num_teachers - 1)
        teacher_target = teacher_probs[:, index, :]
        kd_strength = torch.ones((bsz, 1), device=teacher_probs.device)
        return teacher_target, kd_strength, aux

    if args.method == "average_kd":
        teacher_target = teacher_probs.mean(dim=1)
        kd_strength = torch.ones((bsz, 1), device=teacher_probs.device)
        return teacher_target, kd_strength, aux

    if args.method == "confidence_kd":
        confidence = teacher_probs.max(dim=-1).values  # [B, M]
        weights = F.softmax(confidence / args.weight_temperature, dim=-1)
        teacher_target = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)
        kd_strength = torch.ones((bsz, 1), device=teacher_probs.device)
        aux["teacher_weights"] = weights.detach()
        return teacher_target, kd_strength, aux

    if args.method == "entropy_kd":
        entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=-1)  # [B, M]
        weights = F.softmax((-entropy) / args.weight_temperature, dim=-1)
        teacher_target = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)
        kd_strength = torch.ones((bsz, 1), device=teacher_probs.device)
        aux["teacher_weights"] = weights.detach()
        return teacher_target, kd_strength, aux

    if args.method == "mlp_gating_kd":
        weights = mlp_gate(teacher_probs)
        teacher_target = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)
        kd_strength = torch.ones((bsz, 1), device=teacher_probs.device)
        aux["teacher_weights"] = weights.detach()
        return teacher_target, kd_strength, aux

    if args.method == "dream_kd":
        weights, kd_strength_raw = dream_head(teacher_probs)
        teacher_target = torch.sum(weights.unsqueeze(-1) * teacher_probs, dim=1)

        kd_strength = args.lambda_min + (args.lambda_max - args.lambda_min) * kd_strength_raw

        aux["teacher_weights"] = weights.detach()
        aux["kd_strength"] = kd_strength.detach()
        return teacher_target, kd_strength, aux

    raise ValueError(f"Unsupported method: {args.method}")


def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
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

    data_dir = PROJECT_DIR / args.data_dir
    logit_dir = PROJECT_DIR / "outputs" / args.dataset_name / "teacher_logits"

    run_dir = (
        PROJECT_DIR
        / "outputs"
        / args.dataset_name
        / "students"
        / args.method
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    num_labels = load_num_labels(data_dir, train_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Training student with multi-teacher KD")
    print("=" * 80)
    print(f"Dataset:       {args.dataset_name}")
    print(f"Data dir:      {data_dir}")
    print(f"Logit dir:     {logit_dir}")
    print(f"Run dir:       {run_dir}")
    print(f"Method:        {args.method}")
    print(f"Student model: {args.student_model}")
    print(f"Teachers:      {args.teachers}")
    print(f"Num labels:    {num_labels}")
    print(f"Device:        {device}")

    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=num_labels,
    ).to(device)

    loaders = build_dataloaders(
        data_dir=data_dir,
        logit_dir=logit_dir,
        teachers=args.teachers,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        method=args.method,
    )

    extra_modules = []
    dream_head = None
    mlp_gate = None

    if args.method == "dream_kd":
        dream_head = DREAMKDHead(
            num_teachers=len([t for t in args.teachers if t.lower() != "none"]),
            num_labels=num_labels,
            hidden_dim=args.gate_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        extra_modules.append(dream_head)

    if args.method == "mlp_gating_kd":
        mlp_gate = MLPTeacherGating(
            num_teachers=len([t for t in args.teachers if t.lower() != "none"]),
            num_labels=num_labels,
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

    total_steps = len(loaders["train"]) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
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

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            labels = batch["labels"].to(device)

            teacher_logits = None
            if "teacher_logits" in batch:
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

                if args.method == "student_only" or epoch < args.kd_start_epoch:
                    loss = ce_loss
                    kd_loss = torch.tensor(0.0, device=device)
                else:
                    if teacher_logits is None:
                        raise ValueError("teacher_logits is required for KD methods.")

                    teacher_target, kd_strength, aux = get_teacher_target(
                        args=args,
                        teacher_logits=teacher_logits,
                        dream_head=dream_head,
                        mlp_gate=mlp_gate,
                    )

                    student_log_probs = F.log_softmax(
                        student_logits / args.temperature,
                        dim=-1,
                    )

                    per_sample_kd = F.kl_div(
                        student_log_probs,
                        teacher_target,
                        reduction="none",
                    ).sum(dim=-1) * (args.temperature ** 2)

                    kd_loss = (kd_strength.squeeze(-1) * per_sample_kd).mean()

                    loss = args.ce_weight * ce_loss + args.kd_weight * kd_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())
            total_kd += float(kd_loss.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{ce_loss.item():.4f}",
                kd=f"{kd_loss.item():.4f}",
            )

        dev_metrics = evaluate(model, loaders["dev"], device)
        test_metrics = evaluate(model, loaders["test"], device)

        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(loaders["train"]),
            "train_ce": total_ce / len(loaders["train"]),
            "train_kd": total_kd / len(loaders["train"]),
            "dev": dev_metrics,
            "test": test_metrics,
        }

        history.append(record)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch}")
        print("=" * 80)
        print(f"Dev macro-F1:  {dev_metrics['macro_f1']:.4f}")
        print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]

            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "dev_macro_f1": best_dev_f1,
            }

            if dream_head is not None:
                best_state["dream_head"] = {
                    k: v.detach().cpu()
                    for k, v in dream_head.state_dict().items()
                }

            if mlp_gate is not None:
                best_state["mlp_gate"] = {
                    k: v.detach().cpu()
                    for k, v in mlp_gate.state_dict().items()
                }

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

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
    final_test = evaluate(model, loaders["test"], device)

    final_results = {
        "dataset_name": args.dataset_name,
        "method": args.method,
        "student_model": args.student_model,
        "teachers": args.teachers,
        "num_labels": num_labels,
        "best_dev_macro_f1": best_dev_f1,
        "dev": final_dev,
        "test": final_test,
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
    print("Final results")
    print("=" * 80)
    print(json.dumps(final_results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {run_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--teachers", nargs="+", required=True)

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "student_only",
            "single_kd",
            "average_kd",
            "confidence_kd",
            "entropy_kd",
            "mlp_gating_kd",
            "dream_kd",
        ],
    )

    parser.add_argument("--student_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--single_teacher_index", type=int, default=-1)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--weight_temperature", type=float, default=1.0)

    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--kd_weight", type=float, default=0.5)

    parser.add_argument("--lambda_min", type=float, default=0.2)
    parser.add_argument("--lambda_max", type=float, default=1.0)

    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # KD warmup:
    # kd_start_epoch=1 means KD starts from epoch 1.
    # kd_start_epoch=3 means epochs 1-2 use CE only, epoch 3 starts KD.
    parser.add_argument("--kd_start_epoch", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()