import os
import sys
import argparse
import copy
import json
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics_tracker import MetricsTracker
from src.data.dataset import SEED_DEDataset, SEED_EEGDataset, compute_ea_params
from src.models.eegnet import EEGNet
from src.models.gcenet import GCE_Net
from src.training.trainer import Trainer
from src.training.evaluator import evaluate, evaluate_per_class, voting_accuracy
from src.training.progressive_calibrator import progressive_calibrate
from src.training.prototype_calibrator import prototype_calibrate
from src.training.losses import SupConLoss
from src.utils.checkpoint import save_checkpoint, load_checkpoint

from src.analysis.statistics import full_comparison_report, confidence_interval_95
from src.analysis.error_analysis import per_class_confusion_metrics, hardest_class_pair, error_case_summary

from src.visualization.curves import plot_learning_curves
from src.visualization.confusion import plot_confusion_matrix, plot_paired_confusion_bar
from src.visualization.tsne import plot_tsne
from src.visualization.attention import plot_spatial_attention, plot_se_attention


def build_model(config, A_norm=None):
    model_name = config.get("model_name", "gcenet")
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
            use_gcn=abl.get("use_gcn", True),
            use_dann1=abl.get("use_dann1", True),
            use_dann2=abl.get("use_dann2", True),
            use_supcon=abl.get("use_supcon", False),
            use_spatial_pool=abl.get("use_spatial_pool", True),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_session_map(dataset):
    sid_dates = collections.defaultdict(set)
    for filepath, sid, trial in dataset.samples:
        fname = os.path.basename(filepath)
        date_str = fname.split("_")[1].replace(".mat", "")
        sid_dates[sid].add(date_str)

    session_map = {}
    for sid, dates in sid_dates.items():
        sorted_dates = sorted(dates)
        for session_num, date_str in enumerate(sorted_dates, 1):
            session_map[(sid, date_str)] = session_num

    return session_map


def stratified_split_calibration(dataset, per_class=1, seed=None):
    session_map = _build_session_map(dataset)

    grouped = collections.defaultdict(list)
    for idx in range(len(dataset)):
        _, y, _, _ = dataset[idx]
        filepath = dataset.samples[idx][0]
        fname = os.path.basename(filepath)
        date_str = fname.split("_")[1].replace(".mat", "")
        sid = int(fname.split("_")[0])
        session = session_map[(sid, date_str)]
        grouped[(session, int(y))].append(idx)

    rng = np.random.RandomState(seed)
    cal_tr_idx, cal_val_idx, test_idx = [], [], []

    for (session, emotion), indices in grouped.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if session <= 2:
            n_train = min(per_class, len(shuffled))
            cal_tr_idx.extend(shuffled[:n_train])
            test_idx.extend(shuffled[n_train:])
        else:
            n_val = min(per_class, len(shuffled))
            cal_val_idx.extend(shuffled[:n_val])
            test_idx.extend(shuffled[n_val:])

    return cal_tr_idx, cal_val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--calibrate_only", action="store_true",
                        help="Skip training, load checkpoints and run calibration only")
    parser.add_argument("--resume_dir", default=None,
                        help="Directory with saved checkpoints (default: results/<run_name>/checkpoints)")
    parser.add_argument("--prototype_only", action="store_true",
                        help="Use prototype-based classification instead of LoRA fine-tuning")
    parser.add_argument("--cold_only", action="store_true",
                        help="Skip calibration; train and evaluate cold-start only")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config.get("model_name", "experiment")

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(config["paths"]["results"], run_name), exist_ok=True)
    os.makedirs(os.path.join(config["paths"]["results"], run_name, "figures"), exist_ok=True)

    logger, writer = setup_logger(run_name)
    logger.info(f"Starting cross-validation: {run_name}")

    result_dir = os.path.join(config["paths"]["results"], run_name)

    A_norm = None
    if config["model_name"] == "gcenet":
        adj_path = config["paths"]["adjacency_matrix"]
        if not os.path.exists(adj_path):
            logger.info("Building adjacency matrix...")
            from src.utils.build_graph import build_adjacency
            build_adjacency(save_path=adj_path)
        A_norm = torch.load(adj_path)
        logger.info(f"Loaded adjacency matrix {A_norm.shape} from {adj_path}")

    all_subjects = list(range(1, 16))
    tracker = MetricsTracker()
    best_fold_acc = -1
    best_fold_idx = -1
    best_fold_data = None

    cal_per_class_list = config.get("calibration", {}).get("per_class_ablation", [1, 2, 3, 5])
    cal_tracker = {size: MetricsTracker() for size in cal_per_class_list}

    extracted_dir = config["paths"]["extracted_features"]
    preprocessed_dir = config["paths"]["preprocessed_eeg"]

    ea_params = None
    if config.get("ea", {}).get("enabled", False):
        gc_config = config.get("gcenet", {})
        logger.info("Computing EA parameters for all subjects...")
        ea_params = compute_ea_params(extracted_dir, all_subjects,
                                       target_time=gc_config.get("target_time", 200))
        logger.info(f"EA parameters computed for {len(ea_params)} subjects")

    for fold_idx, target_subject in enumerate(all_subjects):
        source_subjects = [s for s in all_subjects if s != target_subject]
        logger.info(f"Fold {fold_idx + 1}/{len(all_subjects)}: target={target_subject}, source={source_subjects}")

        if config["model_name"] == "eegnet" and config["eegnet"]["input_channels"] == 1:
            raise NotImplementedError("EEGNet with per-subject validation split not yet implemented")
        else:
            target_time = config.get("gcenet", {}).get("target_time", 200)
            dataset_cls = SEED_DEDataset
            dataset_kwargs = dict(data_dir=extracted_dir, target_time=target_time, ea_params=ea_params)

            train_indices, val_indices = [], []
            source_dataset = dataset_cls(extracted_dir, source_subjects, target_time=target_time, ea_params=ea_params)
            for sid in source_subjects:
                sid_indices_by_emotion = {c: [] for c in range(3)}
                for idx in range(len(source_dataset)):
                    filepath, subject_id, _ = source_dataset.samples[idx]
                    if subject_id != sid:
                        continue
                    y = source_dataset[idx][1]
                    sid_indices_by_emotion[y].append(idx)
                rng = np.random.RandomState(config["training"]["seed"] + fold_idx + sid)
                for c in range(3):
                    rng.shuffle(sid_indices_by_emotion[c])
                    for idx in sid_indices_by_emotion[c][2:]:
                        train_indices.append(idx)
                    for idx in sid_indices_by_emotion[c][:2]:
                        val_indices.append(idx)
            train_dataset = torch.utils.data.Subset(source_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(source_dataset, val_indices)
            test_dataset = dataset_cls(extracted_dir, [target_subject], target_time=target_time, ea_params=ea_params)

        num_workers = config["training"].get("num_workers", 0)
        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=num_workers)

        model = build_model(config, A_norm).to(device)

        tc = config["training"]
        lc = config["loss"]
        abl = config["ablation"]

        if args.calibrate_only:
            resume_dir = args.resume_dir or os.path.join(config["paths"]["results"], run_name)
            ckpt_path = os.path.join(resume_dir, "checkpoints", f"fold{fold_idx+1}.pth")
            if not os.path.exists(ckpt_path):
                logger.error(f"Checkpoint not found: {ckpt_path}, skipping fold")
                continue
            load_checkpoint(ckpt_path, model, device=device)
            best_epoch = -1
            best_val_acc = 0.0
            history = {}
        else:
            trainer = Trainer(
                model=model,
                device=device,
                run_name=f"{run_name}_fold{fold_idx+1}",
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
                constant_alpha=tc.get("constant_alpha", None),
                use_ordinal=tc.get("use_ordinal", False),
                ordinal_w2=tc.get("ordinal_w2", 1.5),
            )

            history, best_epoch, best_val_acc = trainer.train(train_loader, val_loader)

            ckpt_dir = os.path.join(result_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(model, None, best_epoch,
                            os.path.join(ckpt_dir, f"fold{fold_idx+1}.pth"))

        test_result, test_preds, test_labels, _ = evaluate(model, test_loader, device)

        fold_metrics = {
            "cold_acc": test_result["accuracy"],
            "cold_f1": test_result["macro_f1"],
            "cold_auc": test_result["macro_auc"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "gap": best_val_acc - test_result["accuracy"],
        }

        if not args.cold_only:
            for per_class in cal_per_class_list:
                cal_tr_idx, cal_val_idx, test_remaining_idx = stratified_split_calibration(
                    test_dataset, per_class=per_class, seed=config["training"]["seed"] + fold_idx
                )

                if len(cal_tr_idx) == 0 or len(cal_val_idx) == 0:
                    logger.warning(
                        f"Fold {fold_idx+1}: per_class={per_class}, insufficient cal samples "
                        f"(train={len(cal_tr_idx)}, val={len(cal_val_idx)}), skipping"
                    )
                    continue

                cal_batch_size = min(2, len(cal_tr_idx))
                cal_train_loader = DataLoader(
                    torch.utils.data.Subset(test_dataset, cal_tr_idx),
                    batch_size=cal_batch_size, shuffle=True,
                )
                cal_val_loader = DataLoader(
                    torch.utils.data.Subset(test_dataset, cal_val_idx),
                    batch_size=config["training"]["batch_size"], shuffle=False,
                )

                if len(test_remaining_idx) == 0:
                    logger.warning(
                        f"Fold {fold_idx+1}: per_class={per_class}, no test samples remaining, skipping"
                    )
                    continue

                test_remaining_loader = DataLoader(
                    torch.utils.data.Subset(test_dataset, test_remaining_idx),
                    batch_size=config["training"]["batch_size"], shuffle=False,
                )

                teacher = copy.deepcopy(model)
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad = False

                if args.prototype_only:
                    cal_result, cal_preds, cal_labels, _ = prototype_calibrate(
                        model, cal_train_loader, test_remaining_loader, device,
                    )
                else:
                    cal_result, cal_preds, cal_labels, cal_probs, _ = progressive_calibrate(
                        model, cal_train_loader, cal_val_loader, test_remaining_loader, device,
                        per_class=per_class,
                        config=config,
                        teacher_model=teacher,
                    )

                fold_metrics[f"cal{per_class}_acc"] = cal_result["accuracy"]
                fold_metrics[f"cal{per_class}_f1"] = cal_result["macro_f1"]
                fold_metrics[f"gap_after_cal{per_class}"] = best_val_acc - cal_result["accuracy"]

                cal_tracker[per_class].add_fold(fold_idx + 1, {
                    "accuracy": cal_result["accuracy"],
                    "macro_f1": cal_result["macro_f1"],
                    "macro_auc": cal_result["macro_auc"],
                })

        tracker.add_fold(fold_idx + 1, fold_metrics)

        if not args.calibrate_only and test_result["accuracy"] > best_fold_acc:
            best_fold_acc = test_result["accuracy"]
            best_fold_idx = fold_idx + 1
            best_fold_data = {
                "history": history,
                "test_preds": test_preds,
                "test_labels": test_labels,
                "fold_idx": best_fold_idx,
            }

        logger.info(f"Fold {fold_idx+1}: cold_acc={test_result['accuracy']:.4f}, "
                     f"best_val={best_val_acc:.4f}, gap={best_val_acc - test_result['accuracy']:.4f}")

    metrics_path = os.path.join(result_dir, "metrics.json")
    tracker.to_json(metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")

    if not args.cold_only:
        for per_class in cal_per_class_list:
            cal_path = os.path.join(result_dir, f"metrics_cal{per_class}.json")
            cal_tracker[per_class].to_json(cal_path)

    if not args.calibrate_only and best_fold_data:
        logger.info(f"Generating visualizations for best fold {best_fold_idx}")
        fig_dir = os.path.join(result_dir, "figures")

        history = best_fold_data["history"]
        plot_learning_curves(history, save_path=os.path.join(fig_dir, "learning_curves.png"))

        cm = plot_confusion_matrix(
            best_fold_data["test_labels"], best_fold_data["test_preds"],
            save_path=os.path.join(fig_dir, "confusion_matrix.png"),
        )
        plot_paired_confusion_bar(cm, save_path=os.path.join(fig_dir, "pairwise_confusion.png"))

        error_summary = error_case_summary(best_fold_data["test_labels"], best_fold_data["test_preds"])
        confusion_pairs = per_class_confusion_metrics(cm)
        hardest = hardest_class_pair(cm)

        analysis_data = {
            "error_analysis": error_summary,
            "confusion_pairs": confusion_pairs,
            "hardest_pair": hardest,
            "best_fold": best_fold_idx,
        }
        with open(os.path.join(result_dir, "analysis.json"), "w") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Error rate: {error_summary['error_rate']:.4f}")
        logger.info(f"Hardest pair: {hardest['pair']} ({hardest['confusion_rate']:.4f})")

    summary = tracker._summarize()
    logger.info("=== Cross-Validation Summary ===")
    for key in sorted(summary):
        if summary[key]["mean"] is not None:
            logger.info(f"  {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    cold_accs = [tracker.folds[str(f)]["cold_acc"] for f in range(1, 16)]
    ci = confidence_interval_95(cold_accs)
    logger.info(f"Cold accuracy 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    writer.close()
    logger.info("Cross-validation complete.")


if __name__ == "__main__":
    main()
