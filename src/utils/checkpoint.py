import os
import torch


def save_checkpoint(model, optimizer, epoch, path, fold=None, is_best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    if fold is not None:
        base, ext = os.path.splitext(path)
        path = f"{base}_fold{fold}{ext}"
    torch.save(checkpoint, path)
    if is_best and fold is not None:
        best_path = path.replace(".pth", "_best.pth")
        torch.save(checkpoint, best_path)


def load_checkpoint(path, model, optimizer=None, device=None):
    checkpoint = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)
