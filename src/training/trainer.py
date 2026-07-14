import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .evaluator import evaluate
from .losses import CombinedLoss, SupConLoss


def _grl_schedule(epoch, total_epochs=50):
    p = epoch / total_epochs
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p)).item()) - 1.0


class Trainer:
    def __init__(
        self,
        model,
        device,
        run_name="default",
        lr=0.001,
        epochs=50,
        early_stop_patience=8,
        early_stop_min_epochs=15,
        lr_factor=0.5,
        lr_patience=5,
        lambda_dann1=0.5,
        lambda_dann2=0.5,
        lambda_supcon=0.3,
        temperature=0.2,
        label_smoothing=0.1,
        use_domain1=True,
        use_domain2=True,
        use_supcon=False,
        use_gcn=True,
        use_spatial_pool=True,
        constant_alpha=None,
        use_ordinal=False,
        ordinal_w2=1.5,
    ):
        self.model = model
        self.device = device
        self.run_name = run_name
        self.epochs = epochs
        self.constant_alpha = constant_alpha
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_epochs = early_stop_min_epochs
        self.use_domain1 = use_domain1
        self.use_domain2 = use_domain2
        self.use_supcon = use_supcon
        self.use_gcn = use_gcn
        self.use_spatial_pool = use_spatial_pool

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=lr_factor, patience=lr_patience
        )
        self.criterion = CombinedLoss(
            lambda_dann1=lambda_dann1,
            lambda_dann2=lambda_dann2,
            lambda_supcon=lambda_supcon,
            temperature=temperature,
            label_smoothing=label_smoothing,
            use_ordinal=use_ordinal,
            ordinal_w2=ordinal_w2,
        )

        self.best_val_acc = 0.0
        self.best_state = None
        self.best_epoch = 0
        self.no_improve = 0
        self.history = {"train_loss": [], "val_acc": [], "domain_acc": [], "lr": []}

    def _forward_batch(self, X, y, sid, alpha):
        if not hasattr(self.model, 'domain1_grl'):
            out = self.model(X)
            return out, None, None, None
        return self.model(
            X,
            alpha=alpha,
            return_all=False,
            use_domain1=self.use_domain1,
            use_domain2=self.use_domain2,
            use_supcon=self.use_supcon,
            use_gcn=self.use_gcn,
            use_spatial_pool=self.use_spatial_pool,
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_domain_correct = 0
        total_domain_samples = 0
        n_batches = 0

        alpha = self.constant_alpha if self.constant_alpha is not None else _grl_schedule(epoch, self.epochs)

        for batch in loader:
            X, y, sid, _ = batch
            X, y, sid = X.to(self.device), y.to(self.device), sid.to(self.device)

            self.optimizer.zero_grad()
            emotion_out, d1_out, d2_out, supcon_out = self._forward_batch(X, y, sid, alpha)

            loss, loss_dict = self.criterion(
                emotion_out, d1_out, d2_out, supcon_out,
                y, sid, sid,
                use_domain1=self.use_domain1,
                use_domain2=self.use_domain2,
                use_supcon=self.use_supcon,
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss_dict["total"]
            if d2_out is not None:
                total_domain_correct += (d2_out.argmax(dim=1) == sid).sum().item()
                total_domain_samples += sid.size(0)
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        domain_acc = total_domain_correct / max(1, total_domain_samples) if total_domain_samples > 0 else 0.0
        return avg_loss, domain_acc, alpha

    def train(self, train_loader, val_loader):
        import logging
        log = logging.getLogger(self.run_name)

        for epoch in range(self.epochs):
            train_loss, domain_acc, alpha = self.train_epoch(train_loader, epoch)

            val_result, _, _, _ = evaluate(self.model, val_loader, self.device)
            val_acc = val_result["accuracy"]

            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_acc"].append(val_acc)
            self.history["domain_acc"].append(domain_acc)
            self.history["lr"].append(current_lr)

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                log.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"loss={train_loss:.4f}, val_acc={val_acc:.4f}, "
                    f"domain_acc={domain_acc:.4f}, lr={current_lr:.6f}"
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                self.no_improve = 0
            else:
                self.no_improve += 1

            if (
                self.no_improve >= self.early_stop_patience
                and epoch >= self.early_stop_min_epochs
            ):
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.history, self.best_epoch, self.best_val_acc
