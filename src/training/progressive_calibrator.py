import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..models.lora import inject_lora_to_emotion_head, remove_lora_from_emotion_head
from .losses import CombinedLoss


def _model_forward(net, X, alpha=0.0, use_domain1=True, use_domain2=True,
                   use_supcon=False, use_gcn=True, use_spatial_pool=True):
    try:
        out = net(X, alpha=alpha, return_all=True,
                  use_domain1=use_domain1, use_domain2=use_domain2,
                  use_supcon=use_supcon, use_gcn=use_gcn,
                  use_spatial_pool=use_spatial_pool)
        if isinstance(out, tuple) and len(out) >= 6:
            return out
        if isinstance(out, tuple):
            return out + (None,) * (6 - len(out))
        return out, None, None, None, None, None
    except (TypeError, AttributeError):
        out = net(X)
        return out, None, None, None, None, None


def _collect_predictions(net, loader, device):
    net.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            X, y, sid, _ = batch
            X, y = X.to(device), y.to(device)
            emotion_out, _, _, _, _, _ = _model_forward(net, X)
            probs = torch.softmax(emotion_out, dim=1)
            preds = emotion_out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    if not all_preds:
        return np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def _metrics_from_predictions(preds, labels, probs):
    if len(preds) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_auc": 0.0}
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return {"accuracy": acc, "macro_f1": f1, "macro_auc": auc}


def _adaptive_cal_lr(base_lr, per_class, max_per_class=5):
    scale = math.sqrt(per_class / max_per_class)
    return base_lr * scale


def _is_gcenet(model):
    abl = getattr(model, "ablation", {})
    return abl.get("use_gcn", False) and hasattr(model, "stage3_conv")


def _freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_lora(model):
    lora = model.emotion_head
    for param in [lora.A, lora.B]:
        param.requires_grad = True
    if lora.lora_bias is not None:
        lora.lora_bias.requires_grad = True


def _unfreeze_stage3(model):
    for param in model.stage3_conv.parameters():
        param.requires_grad = True
    for param in model.se.parameters():
        param.requires_grad = True


def _unfreeze_gcn_spatial(model):
    for name, param in model.gcn.named_parameters():
        param.requires_grad = True
    for name, param in model.spatial_pool.named_parameters():
        param.requires_grad = True
    for name, param in model.residual_proj.named_parameters():
        param.requires_grad = True


def _get_active_stages(model, per_class):
    stages = [1]
    if _is_gcenet(model):
        if per_class >= 2:
            stages.append(2)
        if per_class >= 3:
            stages.append(3)
    return stages


def _apply_stage_unfreeze(model, stage):
    _freeze_all(model)
    _unfreeze_lora(model)
    if stage == 2:
        _unfreeze_stage3(model)
    elif stage == 3:
        _unfreeze_stage3(model)
        _unfreeze_gcn_spatial(model)


def progressive_calibrate(
    model,
    cal_train_loader,
    cal_val_loader,
    test_loader,
    device,
    per_class=1,
    base_lr=0.0003,
    stage_epochs=None,
    stage_patience=None,
    stage_lr_scale=None,
    config=None,
    teacher_model=None,
    distill_alpha=0.3,
    distill_temperature=3.0,
):
    if config is not None:
        cal_cfg = config.get("calibration", {})
        base_lr = config["training"].get("cal_lr", base_lr)
        stage_epochs = cal_cfg.get("stage_epochs", [10, 10, 10])
        stage_patience = cal_cfg.get("stage_patience", [2, 2, 3])
        stage_lr_scale = cal_cfg.get("stage_lr_scale", [1.0, 0.3, 0.1])
        lora_rank = cal_cfg.get("lora_rank", 4)
        lora_alpha = cal_cfg.get("lora_alpha", 1.0)
        aug_noise = cal_cfg.get("aug_noise", 0.0)
        weight_decay = cal_cfg.get("weight_decay", 0.0)
        cal_alpha = cal_cfg.get("cal_alpha", 0.0)
        distill_alpha = cal_cfg.get("distill_alpha", 0.3)
        distill_temperature = cal_cfg.get("distill_temperature", 3.0)
        feature_distill_alpha = cal_cfg.get("feature_distill_alpha", 0.0)
        aug_repeats = cal_cfg.get("aug_repeats", 1)
        loss_cfg = config.get("loss", {})
        lambda_dann1 = loss_cfg.get("lambda_dann1", 0.5)
        lambda_dann2 = loss_cfg.get("lambda_dann2", 0.5)
        label_smoothing = loss_cfg.get("label_smoothing", 0.1)
        ablation = config.get("ablation", {})
        use_domain1 = ablation.get("use_dann1", True)
        use_domain2 = ablation.get("use_dann2", True)
        use_gcn = ablation.get("use_gcn", True)
        use_spatial_pool = ablation.get("use_spatial_pool", True)
    else:
        if stage_epochs is None:
            stage_epochs = [10, 10, 10]
        if stage_patience is None:
            stage_patience = [2, 2, 3]
        if stage_lr_scale is None:
            stage_lr_scale = [1.0, 0.3, 0.1]
        lora_rank = 4
        lora_alpha = 1.0
        aug_noise = 0.0
        weight_decay = 0.0
        cal_alpha = 0.0
        distill_alpha = 0.3
        distill_temperature = 3.0
        feature_distill_alpha = 0.0
        aug_repeats = 1
        lambda_dann1 = 0.5
        lambda_dann2 = 0.5
        label_smoothing = 0.1
        use_domain1 = True
        use_domain2 = True
        use_gcn = True
        use_spatial_pool = True

    cal_net = copy.deepcopy(model)
    cal_net = inject_lora_to_emotion_head(cal_net, rank=lora_rank, alpha=lora_alpha)
    cal_net.train()

    domain_map = None
    if (use_domain1 and lambda_dann1 > 0) or (use_domain2 and lambda_dann2 > 0):
        unique_sids = set()
        for batch in cal_train_loader:
            _, _, sid_batch, _ = batch
            unique_sids.update(sid_batch.tolist())
        domain_map = {orig: i for i, orig in enumerate(sorted(unique_sids))}

    active_stages = _get_active_stages(cal_net, per_class)
    stage_lr = _adaptive_cal_lr(base_lr, per_class)

    for stage_idx, stage in enumerate(active_stages):
        _apply_stage_unfreeze(cal_net, stage)

        lr = stage_lr * stage_lr_scale[stage - 1]
        epochs = stage_epochs[stage - 1]

        trainable_params = [p for p in cal_net.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        criterion = CombinedLoss(
            lambda_dann1=lambda_dann1 if use_domain1 else 0.0,
            lambda_dann2=lambda_dann2 if use_domain2 else 0.0,
            label_smoothing=label_smoothing,
        )

        for epoch in range(epochs):
            cal_net.train()
            for batch in cal_train_loader:
                X, y, sid, _ = batch
                X, y, sid = X.to(device), y.to(device), sid.to(device)
                if domain_map is not None:
                    sid = torch.tensor([domain_map[s.item()] for s in sid.cpu()],
                                       device=device, dtype=torch.long)

                for rep in range(aug_repeats):
                    X_aug = X
                    if aug_noise > 0:
                        X_aug = X + torch.randn_like(X) * aug_noise

                    optimizer.zero_grad()
                    emotion_out, d1_out, d2_out, supcon_out, features, _ = _model_forward(
                        cal_net, X_aug, alpha=cal_alpha,
                        use_domain1=use_domain1, use_domain2=use_domain2,
                        use_gcn=use_gcn, use_spatial_pool=use_spatial_pool,
                    )
                    loss, loss_dict = criterion(
                        emotion_out, d1_out, d2_out, supcon_out,
                        y, sid, sid,
                        use_domain1=use_domain1,
                        use_domain2=use_domain2,
                        use_supcon=False,
                    )

                    if teacher_model is not None and distill_alpha > 0:
                        cal_net.eval()
                        with torch.no_grad():
                            teacher_emo, _, _, _, _, _ = _model_forward(
                                teacher_model, X_aug, alpha=0.0,
                                use_domain1=use_domain1, use_domain2=use_domain2,
                                use_gcn=use_gcn, use_spatial_pool=use_spatial_pool,
                            )
                        cal_net.train()
                        T = distill_temperature
                        student_log_prob = F.log_softmax(emotion_out / T, dim=1)
                        teacher_prob = F.softmax(teacher_emo / T, dim=1)
                        kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T * T)
                        loss = (1.0 - distill_alpha) * loss + distill_alpha * kl_loss

                    if teacher_model is not None and feature_distill_alpha > 0:
                        cal_net.eval()
                        with torch.no_grad():
                            _, _, _, _, teacher_feat, _ = _model_forward(
                                teacher_model, X_aug, alpha=0.0,
                                use_domain1=use_domain1, use_domain2=use_domain2,
                                use_gcn=use_gcn, use_spatial_pool=use_spatial_pool,
                            )
                        cal_net.train()
                        mse_loss = F.mse_loss(features, teacher_feat)
                        loss = (1.0 - feature_distill_alpha) * loss + feature_distill_alpha * mse_loss

                    loss.backward()
                    optimizer.step()

    preds, labels, probs = _collect_predictions(cal_net, test_loader, device)
    cal_result = _metrics_from_predictions(preds, labels, probs)

    return cal_result, preds, labels, probs, cal_net
