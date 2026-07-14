import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, subject_ids):
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        subject_ids = subject_ids.contiguous().view(-1, 1)

        same_emotion = labels.eq(labels.T).float()
        diff_subject = (~subject_ids.eq(subject_ids.T)).float()
        pos_mask = same_emotion * diff_subject
        pos_mask.fill_diagonal_(0)

        valid_anchors = pos_mask.sum(dim=1) > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob_pos[valid_anchors].mean()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_target * log_probs).sum(dim=-1).mean()
        return loss


class CombinedLoss:
    def __init__(
        self,
        lambda_dann1=0.5,
        lambda_dann2=0.5,
        lambda_supcon=0.3,
        temperature=0.2,
        label_smoothing=0.1,
        use_ordinal=False,
        ordinal_w2=1.5,
    ):
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.domain_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.lambda_dann1 = lambda_dann1
        self.lambda_dann2 = lambda_dann2
        self.lambda_supcon = lambda_supcon
        self.use_ordinal = use_ordinal
        self.ordinal_w2 = ordinal_w2

    def __call__(
        self,
        emotion_out,
        domain1_out,
        domain2_out,
        supcon_out,
        emotion_labels,
        domain_labels,
        subject_ids,
        use_domain1=True,
        use_domain2=True,
        use_supcon=False,
    ):
        if self.use_ordinal:
            logit_neg = emotion_out[:, 0]
            logit_neu = emotion_out[:, 1]
            logit_pos = emotion_out[:, 2]
            task1_logit = torch.logsumexp(
                torch.stack([logit_pos, logit_neu], dim=1), dim=1
            ) - logit_neg
            task1_target = (emotion_labels >= 1).float()
            task2_logit = logit_pos - torch.logsumexp(
                torch.stack([logit_neu, logit_neg], dim=1), dim=1
            )
            task2_target = (emotion_labels == 2).float()
            total = self.bce_loss(task1_logit, task1_target) + \
                    self.ordinal_w2 * self.bce_loss(task2_logit, task2_target)
            losses = {"ce": total.item()}
        else:
            total = self.ce_loss(emotion_out, emotion_labels)
            losses = {"ce": total.item()}

        if use_domain1 and domain1_out is not None and self.lambda_dann1 > 0:
            d1 = self.domain_loss(domain1_out, domain_labels) * self.lambda_dann1
            total = total + d1
            losses["dann1"] = d1.item()

        if use_domain2 and domain2_out is not None and self.lambda_dann2 > 0:
            d2 = self.domain_loss(domain2_out, domain_labels) * self.lambda_dann2
            total = total + d2
            losses["dann2"] = d2.item()

        if use_supcon and supcon_out is not None and self.lambda_supcon > 0:
            sc = self.supcon_loss(supcon_out, emotion_labels, subject_ids) * self.lambda_supcon
            total = total + sc
            losses["supcon"] = sc.item()

        losses["total"] = total.item()
        return total, losses
