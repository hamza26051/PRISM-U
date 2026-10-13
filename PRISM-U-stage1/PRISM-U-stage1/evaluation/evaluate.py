import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss

from data.datasets import get_train_subset, get_test_dataset, make_loader
from models.resnet import build_resnet18_cifar
from utils.io import load_yaml, save_json
from utils.reproducibility import seed_everything


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(probs, labels):
    preds = probs.argmax(1)
    conf = probs.max(1)
    eps = 1e-12
    entropy = -(probs * np.log(probs + eps)).sum(1)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "cross_entropy": float(log_loss(labels, probs, labels=list(range(probs.shape[1])))),
        "mean_confidence": float(conf.mean()),
        "mean_entropy": float(entropy.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cifar10_stage1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_resnet18_cifar(ckpt.get("num_classes", 10)).to(device)
    model.load_state_dict(ckpt["model_state"])

    datasets = {
        "retain": get_train_subset(dcfg["root"], dcfg["split_file"], "retain", augment=False),
        "forget": get_train_subset(dcfg["root"], dcfg["split_file"], "forget", augment=False),
        "auxiliary": get_train_subset(dcfg["root"], dcfg["split_file"], "auxiliary", augment=False),
        "validation": get_train_subset(dcfg["root"], dcfg["split_file"], "validation", augment=False),
        "test": get_test_dataset(dcfg["root"]),
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_mode": ckpt.get("mode"),
        "checkpoint_seed": ckpt.get("seed"),
        "sets": {},
    }
    prob_dump = {}

    for name, ds in datasets.items():
        loader = make_loader(ds, cfg["evaluation"]["batch_size"], False, dcfg["num_workers"])
        probs, labels = collect(model, loader, device)
        report["sets"][name] = metrics(probs, labels)
        print(name, report["sets"][name])
        if cfg["evaluation"].get("save_probabilities", True):
            prob_dump[f"{name}_probs"] = probs
            prob_dump[f"{name}_labels"] = labels

    out_name = args.name or Path(args.checkpoint).stem
    result_dir = Path(cfg["paths"]["results"])
    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, result_dir / f"eval_{out_name}.json")
    if prob_dump:
        np.savez_compressed(result_dir / f"probs_{out_name}.npz", **prob_dump)
    print(f"Saved evaluation to {result_dir}")


if __name__ == "__main__":
    main()
