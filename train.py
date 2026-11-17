from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model, prepare_image


SPLITS = {
    "train": ("Training", "training_words", "training_labels.csv"),
    "validation": ("Validation", "validation_words", "validation_labels.csv"),
    "test": ("Testing", "testing_words", "testing_labels.csv"),
}


class PrescriptionDataset(Dataset):
    def __init__(
        self,
        records: list[tuple[Path, str]],
        class_to_idx: dict[str, int],
        image_transform: transforms.Compose,
        image_size: int | tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.transform = image_transform
        self.image_size = image_size
        self.samples = [(path, class_to_idx[medicine]) for path, medicine in records]
        if not self.samples:
            raise ValueError("No labeled images found in this split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = prepare_image(image, self.image_size)
        return self.transform(image), label


def read_training_labels(data_dir: Path) -> tuple[list[str], dict[str, str]]:
    labels_path = data_dir / SPLITS["train"][0] / SPLITS["train"][2]
    with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    classes = sorted({row["MEDICINE_NAME"].strip() for row in rows})
    generic_names: dict[str, str] = {}
    for row in rows:
        medicine = row["MEDICINE_NAME"].strip()
        generic = row["GENERIC_NAME"].strip()
        if medicine in generic_names and generic_names[medicine] != generic:
            raise ValueError(f"Conflicting generic names for medicine {medicine!r}")
        generic_names[medicine] = generic
    if len(classes) < 2:
        raise ValueError("Training labels must contain at least two medicine classes.")
    return classes, generic_names


def load_unique_split_records(
    data_dir: Path,
) -> tuple[dict[str, list[tuple[Path, str]]], dict[str, int], dict[str, int]]:
    records: dict[str, list[tuple[Path, str]]] = {}
    hash_labels: dict[bytes, set[str]] = {}
    for split, (directory, image_dir, labels_file) in SPLITS.items():
        labels_path = data_dir / directory / labels_file
        image_root = data_dir / directory / image_dir
        split_records: list[tuple[Path, str, bytes]] = []
        with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                image_path = image_root / row["IMAGE"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"Image listed in labels is missing: {image_path}")
                medicine = row["MEDICINE_NAME"].strip()
                digest = hashlib.sha256(image_path.read_bytes()).digest()
                hash_labels.setdefault(digest, set()).add(medicine)
                split_records.append((image_path, medicine, digest))
        records[split] = split_records

    kept = {split: [] for split in SPLITS}
    excluded = {split: 0 for split in SPLITS}
    ambiguous_excluded = {split: 0 for split in SPLITS}
    conflicting_hashes = {
        digest for digest, labels in hash_labels.items() if len(labels) > 1
    }
    seen: set[bytes] = set()
    for split in ("test", "validation", "train"):
        for path, medicine, digest in records[split]:
            if digest in conflicting_hashes:
                excluded[split] += 1
                ambiguous_excluded[split] += 1
                continue
            if digest in seen:
                excluded[split] += 1
                continue
            seen.add(digest)
            kept[split].append((path, medicine))
    return kept, excluded, ambiguous_excluded


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a ResNet50 medicine-word classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root folder containing Training, Validation, and Testing.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continue from the checkpoint in --output-dir.")
    parser.add_argument("--no-pretrained", action="store_true", help="Train without downloading ImageNet weights.")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive.")
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"Dataset folder does not exist: {data_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes, generic_names = read_training_labels(data_dir)
    class_to_idx = {name: index for index, name in enumerate(classes)}
    split_records, duplicate_counts, ambiguous_counts = load_unique_split_records(data_dir)
    for split, records in split_records.items():
        present_classes = {medicine for _, medicine in records}
        if present_classes != set(classes):
            raise ValueError(f"The {split} split is missing one or more training classes after deduplication.")
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=5,
                translate=(0.04, 0.04),
                scale=(0.9, 1.1),
                fill=(255, 255, 255),
            ),
            transforms.ColorJitter(brightness=0.12, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    datasets = {
        "train": PrescriptionDataset(
            split_records["train"], class_to_idx, train_transform, IMAGE_SIZE
        ),
        "validation": PrescriptionDataset(
            split_records["validation"], class_to_idx, eval_transform, IMAGE_SIZE
        ),
        "test": PrescriptionDataset(
            split_records["test"], class_to_idx, eval_transform, IMAGE_SIZE
        ),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        for split, dataset in datasets.items()
    }
    print(
        f"Loaded {len(classes)} classes: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items())
    )
    print(f"Exact duplicate images excluded by split: {duplicate_counts}")
    print(f"Conflicting-label image copies excluded by split: {ambiguous_counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "resnet50_medicine.pt"
    if args.resume and not checkpoint_path.is_file():
        parser.error(f"Cannot resume because the checkpoint does not exist: {checkpoint_path}")

    start_epoch = 1
    best_accuracy = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["classes"] != classes:
            parser.error("The checkpoint medicine classes do not match the dataset.")
        model = build_model(len(classes)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint.get("epoch", 1)) + 1
        if start_epoch > args.epochs:
            parser.error("--epochs must be greater than the checkpoint's saved epoch when resuming.")
        best_accuracy, best_loss = accuracy(model, loaders["validation"], device)
        print(
            f"Resuming from epoch {start_epoch - 1}; "
            f"current validation accuracy={best_accuracy:.2%}"
        )
    else:
        model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("layer3.", "layer4.", "fc."))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer3.parameters(), "lr": 2e-5},
            {"params": model.layer4.parameters(), "lr": 6e-5},
            {"params": model.fc.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        running_loss = 0.0
        seen = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_accuracy, val_loss = accuracy(model, loaders["validation"], device)
        scheduler.step(val_accuracy)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={running_loss / seen:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.2%}"
        )

        improved = val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_loss
        )
        if improved:
            best_accuracy, best_loss = val_accuracy, val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "generic_names": generic_names,
                    "image_size": list(IMAGE_SIZE),
                    "validation_accuracy": best_accuracy,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation score did not improve.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_accuracy, test_loss = accuracy(model, loaders["test"], device)
    results = {
        "classes": len(classes),
        "training_images": len(datasets["train"]),
        "validation_images": len(datasets["validation"]),
        "test_images": len(datasets["test"]),
        "duplicate_images_excluded": duplicate_counts,
        "conflicting_label_images_excluded": ambiguous_counts,
        "epochs_completed": last_epoch,
        "training_complete": last_epoch >= args.epochs,
        "best_validation_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "checkpoint": checkpoint_path.name,
    }
    results_path = args.output_dir / "evaluation.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Held-out test accuracy: {test_accuracy:.2%}")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved evaluation: {results_path}")


if __name__ == "__main__":
    main()
