"""Reload the saved checkpoint in a fresh process and confirm it still scores the same.

Usage:
    python verify.py
    python verify.py --ckpt ./checkpoints/resnet18_cifar10.pt --data-dir ./data          # laptop, full test set
    python verify.py --ckpt ./checkpoints/resnet18_cifar10.pt --data-dir ./sample-data   # laptop, 20 PNGs only

Exits with code 1 if the reloaded accuracy differs from the saved one by more than --tolerance.
If --data-dir holds PNG files, it scores those instead (no download) and only checks accuracy >= 70%.
"""

import argparse
import glob
import os
import sys

import torch
import torchvision
from PIL import Image

from train import build_model, evaluate, test_transform


def verify_samples(model, classes, sample_dir, device, min_acc=0.7):
    """Score a folder of PNGs named <id>_<class>.png (e.g. sample-data/), no dataset download needed."""
    paths = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    tf = test_transform()
    images = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    labels = [os.path.splitext(os.path.basename(p))[0].split("_", 1)[1] for p in paths]

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(images.to(device).to(memory_format=torch.channels_last)), dim=1).cpu()

    correct = 0
    for path, label, p in zip(paths, labels, probs):
        pred = classes[p.argmax().item()]
        correct += pred == label
        mark = "ok " if pred == label else "MISS"
        print(f"  {mark}  {os.path.basename(path):<22} predicted {pred:<10} ({p.max().item():.0%})  actual {label}")

    acc = correct / len(paths)
    print(f"Sample accuracy: {correct}/{len(paths)} ({acc:.0%})")
    # 20 images is too few to match the saved test accuracy exactly; this only catches a broken reload
    # (an untrained or mis-loaded model scores around 10%).
    if acc < min_acc:
        print(f"FAIL: below {min_acc:.0%} on the samples, the checkpoint did not load correctly.")
        sys.exit(1)
    print("PASS: checkpoint reloads and classifies the samples correctly.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="/workspace/checkpoints/resnet18_cifar10.pt")
    p.add_argument("--data-dir", default="/workspace/data")
    p.add_argument("--tolerance", type=float, default=0.005, help="allowed accuracy difference")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"Checkpoint trained on {ckpt['device']} with torch {ckpt['torch_version']}")

    model = build_model(num_classes=len(ckpt["classes"])).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])

    if glob.glob(os.path.join(args.data_dir, "*.png")):
        verify_samples(model, ckpt["classes"], args.data_dir, device)
        return

    test_set = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform())
    if ckpt.get("subset"):
        test_set = torch.utils.data.Subset(test_set, range(min(ckpt["subset"], len(test_set))))
    loader = torch.utils.data.DataLoader(test_set, batch_size=512, num_workers=2)

    acc = evaluate(model, loader, device, use_amp=device.type == "cuda")
    saved = ckpt["test_acc"]
    print(f"Saved accuracy:    {saved:.2%}")
    print(f"Reloaded accuracy: {acc:.2%}  (on {device.type})")

    images, labels = next(iter(loader))
    with torch.no_grad():
        preds = model(images[:8].to(device).to(memory_format=torch.channels_last)).argmax(1).cpu()
    for pred, label in zip(preds, labels[:8]):
        mark = "ok " if pred == label else "MISS"
        print(f"  {mark}  predicted {ckpt['classes'][pred]:<10}  actual {ckpt['classes'][label]}")

    if abs(acc - saved) > args.tolerance:
        print("FAIL: reloaded accuracy does not match the saved checkpoint.")
        sys.exit(1)
    print("PASS: checkpoint reloads and matches.")


if __name__ == "__main__":
    main()
