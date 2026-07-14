import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _extract_features(net, loader, device):
    net.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X, y, sid, _ = batch
            X, y = X.to(device), y.to(device)
            try:
                _, _, _, _, features, _ = net(X, alpha=0.0, return_all=True)
            except (TypeError, AttributeError):
                out = net(X)
                features = out
            all_features.append(features.cpu())
            all_labels.append(y.cpu())
    if not all_features:
        return None, None
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def prototype_calibrate(model, cal_train_loader, test_loader, device):
    cal_features, cal_labels = _extract_features(model, cal_train_loader, device)
    if cal_features is None:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_auc": 0.0}, [], [], None

    prototypes = {}
    for c in range(3):
        mask = cal_labels == c
        if mask.sum() > 0:
            prototypes[c] = cal_features[mask].mean(dim=0)

    test_features, test_labels = _extract_features(model, test_loader, device)
    if test_features is None:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_auc": 0.0}, [], [], None

    proto_matrix = torch.stack([prototypes[c].to(device) for c in range(3)])
    dists = torch.cdist(test_features.to(device), proto_matrix, p=2)
    preds = dists.argmin(dim=1).cpu().numpy()
    labels = test_labels.numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    try:
        probs = F.softmax(-dists, dim=1).cpu().numpy()
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {"accuracy": acc, "macro_f1": f1, "macro_auc": auc}, preds, labels, probs
