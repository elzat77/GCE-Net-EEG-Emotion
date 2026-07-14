import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def _to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [_to_device(d, device) for d in data]
    return data.to(device)


def _model_forward(net, X, alpha=0.0, return_all=False):
    try:
        out = net(X, alpha=alpha, return_all=True)
        if isinstance(out, tuple) and len(out) >= 6:
            return out
        if isinstance(out, tuple):
            return out + (None,) * (6 - len(out))
        return out, None, None, None, None, None
    except (TypeError, AttributeError):
        out = net(X)
        return out, None, None, None, None, None


def evaluate(net, loader, device, domain_labels=None):
    net.eval()
    all_preds, all_labels = [], []
    all_domain_preds, all_domain_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            X, y, sid, _ = batch
            X, y = X.to(device), y.to(device)

            emotion_out, d1_out, d2_out, _, _, _ = _model_forward(net, X)
            probs = torch.softmax(emotion_out, dim=1)
            preds = emotion_out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if domain_labels is not None and d2_out is not None:
                all_domain_preds.append(d2_out.argmax(dim=1).cpu().numpy())
                all_domain_labels.append(sid.cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_probs = np.concatenate(all_probs) if all_probs else np.array([])

    if len(all_preds) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_auc": 0.0}, all_preds, all_labels, all_probs

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    result = {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_auc": auc,
    }

    if all_domain_preds:
        result["domain_acc"] = accuracy_score(
            np.concatenate(all_domain_labels),
            np.concatenate(all_domain_preds),
        )

    return result, all_preds, all_labels, all_probs


def evaluate_per_class(net, loader, device):
    net.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            X, y, sid, trial = batch
            X, y = X.to(device), y.to(device)
            emotion_out, _, _, _, _, _ = _model_forward(net, X)
            preds = emotion_out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    per_class = {}
    for cls in range(3):
        per_class[f"precision_{cls}"] = precision_score(all_labels, all_preds, labels=[cls], average="macro", zero_division=0)
        per_class[f"recall_{cls}"] = recall_score(all_labels, all_preds, labels=[cls], average="macro", zero_division=0)
        per_class[f"f1_{cls}"] = f1_score(all_labels, all_preds, labels=[cls], average="macro", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    return per_class, cm, all_preds, all_labels


def calibrate_and_evaluate(
    net,
    cal_train_loader,
    cal_val_loader,
    test_loader,
    device,
    n_cal_samples=3,
    lr=0.0003,
    epochs=30,
    patience=3,
):
    cal_net = copy.deepcopy(net)
    cal_net.train()

    for param in cal_net.parameters():
        param.requires_grad = False
    for param in cal_net.emotion_head.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, cal_net.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        cal_net.train()
        for batch in cal_train_loader:
            X, y, sid, _ = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            emotion_out, _, _, _, _, _ = _model_forward(cal_net, X)
            loss = criterion(emotion_out, y)
            loss.backward()
            optimizer.step()

        result, _, _, _ = evaluate(cal_net, cal_val_loader, device)
        val_acc = result["accuracy"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(cal_net.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and epoch > patience * 2:
            break

    if best_state is not None:
        cal_net.load_state_dict(best_state)

    result, preds, labels, _ = evaluate(cal_net, test_loader, device)
    return result, preds, labels, cal_net


def voting_accuracy(preds_by_trial, labels_by_trial, trials):
    u_trials = np.unique(trials)
    correct = 0
    total = 0
    for t in u_trials:
        mask = trials == t
        t_preds = preds_by_trial[mask]
        t_label = labels_by_trial[mask][0]
        vote = np.bincount(t_preds).argmax()
        if vote == t_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0
