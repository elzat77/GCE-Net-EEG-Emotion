import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.utils.config import load_config, merge_cli_overrides
from src.utils.logger import setup_logger
from src.utils.checkpoint import save_checkpoint
from src.data.dataset import SEED_DEDataset, SEED_EEGDataset
from src.data.sampler import MultiSubjectBatchSampler
from src.models.eegnet import EEGNet
from src.models.gcenet import GCE_Net
from src.training.trainer import Trainer
from src.training.evaluator import evaluate


def build_model(config, A_norm=None):
    model_name = config["model_name"]
    mc = config["model"]
    tc = config["training"]

    if model_name == "eegnet":
        ec = config["eegnet"]
        return EEGNet(
            n_classes=mc["n_classes"],
            input_channels=ec["input_channels"],
            input_time=ec["input_time"],
            F1=ec["F1"],
            D=ec["D"],
            F2=ec["F2"],
            dropout=mc["dropout"],
        )
    elif model_name == "gcenet":
        gc = config["gcenet"]
        abl = config["ablation"]
        return GCE_Net(
            A_norm=A_norm,
            n_classes=mc["n_classes"],
            n_domains=mc["n_domains"],
            in_channels=gc["in_channels"],
            input_time=gc["input_time"],
            num_electrodes=gc["num_electrodes"],
            temp_filters=gc["temp_filters"],
            gcn_hidden=gc["gcn_hidden"],
            stage3_channels=gc["stage3_channels"],
            se_reduction=gc["se_reduction"],
            supcon_dim=gc["supcon_dim"],
            dropout=mc["dropout"],
            dropedge_p=mc["dropedge_p"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model_name"] = args.model

    set_seed(config["training"]["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    logger, writer = setup_logger(f"fold_{args.fold}")

    A_norm = None
    if config["model_name"] == "gcenet":
        adj_path = config["paths"]["adjacency_matrix"]
        if os.path.exists(adj_path):
            A_norm = torch.load(adj_path)
        else:
            logger.warning(f"Adjacency matrix not found at {adj_path}, will build now")
            from src.utils.build_graph import build_adjacency
            build_adjacency(save_path=adj_path)
            A_norm = torch.load(adj_path)

    all_subjects = list(range(1, 16))
    target_subject = all_subjects[args.fold - 1]
    source_subjects = [s for s in all_subjects if s != target_subject]

    extracted_dir = os.path.join(config["data_path"], "ExtractedFeatures")
    preprocessed_dir = os.path.join(config["data_path"], "Preprocessed_EEG")

    if config["model_name"] == "eegnet" and config["eegnet"]["input_channels"] == 1:
        train_dataset = SEED_EEGDataset(preprocessed_dir, source_subjects)
        val_subject = source_subjects[-1]
        val_dataset = SEED_EEGDataset(preprocessed_dir, [val_subject])
        test_dataset = SEED_EEGDataset(preprocessed_dir, [target_subject])
    else:
        target_time = config.get("gcenet", {}).get("target_time", 200)
        train_dataset = SEED_DEDataset(extracted_dir, source_subjects[:-1], target_time=target_time)
        val_dataset = SEED_DEDataset(extracted_dir, source_subjects[-1:], target_time=target_time)
        test_dataset = SEED_DEDataset(extracted_dir, [target_subject], target_time=target_time)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    model = build_model(config, A_norm).to(device)

    tc = config["training"]
    lc = config["loss"]
    abl = config["ablation"]

    trainer = Trainer(
        model=model,
        device=device,
        run_name=f"fold_{args.fold}",
        lr=tc["lr"],
        epochs=tc["epochs"],
        early_stop_patience=tc["early_stop_patience"],
        early_stop_min_epochs=tc["early_stop_min_epochs"],
        lr_factor=tc["lr_factor"],
        lr_patience=tc["lr_patience"],
        lambda_dann1=lc["lambda_dann1"],
        lambda_dann2=lc["lambda_dann2"],
        lambda_supcon=lc["lambda_supcon"],
        temperature=lc["temperature"],
        label_smoothing=lc["label_smoothing"],
        use_domain1=abl.get("use_dann1", True),
        use_domain2=abl.get("use_dann2", True),
        use_supcon=abl.get("use_supcon", False),
        use_gcn=abl.get("use_gcn", True),
        use_spatial_pool=abl.get("use_spatial_pool", True),
    )

    history, best_epoch, best_val_acc = trainer.train(train_loader, val_loader)
    logger.info(f"Fold {args.fold}: best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")

    ckpt_path = os.path.join(config["paths"]["checkpoints"], f"fold_{args.fold}_best.pth")
    save_checkpoint(model, trainer.optimizer, best_epoch, ckpt_path, fold=args.fold, is_best=True)

    test_result, preds, labels, _ = evaluate(model, test_loader, device)
    logger.info(f"Fold {args.fold} test: acc={test_result['accuracy']:.4f}, f1={test_result['macro_f1']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
