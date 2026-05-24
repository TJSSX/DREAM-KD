from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import compute_classification_metrics


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a batch to device."""
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0
    total_steps = 0

    progress = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model and return loss plus metrics."""
    model.eval()

    total_loss = 0.0
    total_steps = 0

    all_labels = []
    all_preds = []

    progress = tqdm(dataloader, desc="Evaluating", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)

        total_loss += loss.item()
        total_steps += 1

        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_steps, 1)
    metrics = compute_classification_metrics(
        y_true=np.asarray(all_labels),
        y_pred=np.asarray(all_preds),
    )

    return avg_loss, metrics