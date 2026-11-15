"""Train a ResNet-18 image classifier on CIFAR-10 on a single GPU.

Standard single-GPU speed-ups used here:
  - mixed precision (fp16 autocast + GradScaler)
  - channels_last memory layout for conv-heavy models
  - cudnn.benchmark to pick the fastest convolution kernels
  - parallel, pinned-memory data loading

Usage (on the pod):
    python train.py                      # 15 epochs, defaults below
    python train.py --epochs 1 --subset 2000   # quick smoke test
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def build_model(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for 32x32 images: 3x3 stem, no initial max-pool."""
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def test_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def make_loaders(data_dir, batch_size, workers, subset, pin):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform())
    if subset:
        train_set = torch.utils.data.Subset(train_set, range(min(subset, len(train_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(subset, len(test_set))))

    common = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.2, help="peak learning rate for OneCycle")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--data-dir", default="/workspace/data")
    p.add_argument("--out-dir", default="/workspace/checkpoints")
    p.add_argument("--subset", type=int, default=0, help="use only N images (smoke test)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  AMP: {use_amp}")
    else:
        print("WARNING: no GPU found, running on CPU (slow). Use --subset for a smoke test.")

    train_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.workers, args.subset, pin=device.type == "cuda")

    model = build_model().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    run_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        loss_sum = correct = seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0
        test_acc = evaluate(model, test_loader, device, use_amp)

        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / seen, 4),
            "train_acc": round(correct / seen, 4),
            "test_acc": round(test_acc, 4),
            "epoch_time_s": round(epoch_time, 1),
            "images_per_s": round(seen / epoch_time),
        }
        if device.type == "cuda":
            row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        history.append(row)
        print(json.dumps(row))

    total_time = time.time() - run_start
    final_acc = history[-1]["test_acc"]
    print(f"Done in {total_time / 60:.1f} min  |  final test accuracy {final_acc:.2%}")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "resnet18_cifar10.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "test_acc": final_acc,
        "subset": args.subset,
        "args": vars(args),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }, ckpt_path)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"total_time_s": round(total_time, 1), "history": history}, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
