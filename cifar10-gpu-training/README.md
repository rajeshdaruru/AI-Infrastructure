# 2. Train on a Rented GPU (CIFAR-10 / ResNet-18)

Rent a cloud GPU, confirm the driver and PyTorch can see it, run a real training job on it, and prove the saved model reloads correctly.

| File | What it does |
|------|--------------|
| [train.py](train.py) | Trains ResNet-18 on CIFAR-10 with mixed precision, channels_last, cudnn.benchmark and parallel data loading. Saves a checkpoint and per-epoch metrics. |
| [verify.py](verify.py) | Reloads the checkpoint in a fresh process, re-scores the test set and checks it matches the saved accuracy. |

Expected on an RTX 4090: about 15 epochs in 5–10 minutes, ~93% test accuracy, GPU utilization at 80–100%.

## Task 1: Rent a GPU and confirm you can see it

1. Create an SSH key (skip if `~/.ssh/id_ed25519.pub` exists) and copy the public key:
   ```bash
   ssh-keygen -t ed25519 -C "gpu-lesson"
   cat ~/.ssh/id_ed25519.pub
   ```
2. On [RunPod](https://www.runpod.io): add ~$10 credit, then paste the key under **Settings → SSH Public Keys**.
3. **Pods → + Deploy → GPU Cloud**: RTX 4090 (or A10/A40), template **RunPod PyTorch 2.x**, container disk 20 GB, volume 20 GB at `/workspace`, **SSH over exposed TCP** on, **On-Demand** (not spot).
4. Connect with the "SSH over exposed TCP" command (needed for `scp`):
   ```bash
   ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
   ```
5. Check the driver: `nvidia-smi` should show the driver/CUDA version, the GPU name and memory, 0% utilization and no running processes.
6. Check PyTorch: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should print `True NVIDIA GeForce RTX 4090`. If it prints `False`, check `pip list | grep torch`; a `+cpu` build needs reinstalling with CUDA support.
7. Save proof: `nvidia-smi > /workspace/task1_nvidia_smi.txt`

## Task 2: Run the training job

From your laptop, copy the scripts to the pod:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 train.py verify.py root@<POD_IP>:/workspace/
```

On the pod, do a quick smoke test, then the full run:
```bash
cd /workspace
python train.py --epochs 1 --subset 2000
python train.py 2>&1 | tee task2_train_log.txt
```

In a second SSH session, watch the GPU while it trains:
```bash
watch -n 1 nvidia-smi
```
GPU-Util should stay high and memory use should jump well above idle. If utilization is low, the data loader is the bottleneck: try `--workers` with a higher number.

### Tuning `--workers`

`--workers` sets how many CPU processes load and prepare images in parallel while the GPU trains. The default is 8, or the number of CPU cores if that is lower.

```bash
python train.py --workers 12 2>&1 | tee task2_train_log.txt
```

1. Check the pod's CPU count with `nproc`. If it reports far more than the vCPUs shown on RunPod's deploy page, it is counting the whole host, so use the RunPod number.
2. Start at roughly that number, or a little below it (e.g. 12 on a 16-vCPU pod).
3. Compare 1-epoch runs and keep the setting with the highest `images_per_s`:
   ```bash
   python train.py --epochs 1 --workers 4
   python train.py --epochs 1 --workers 12
   ```

| `nvidia-smi` shows | Meaning | Action |
|--------------------|---------|--------|
| GPU-Util steady at 90–100% | Data loading keeps up | More workers won't help |
| GPU-Util jumps between high and low | GPU is waiting for data | Raise `--workers` |
| Slower after raising workers above the core count | Workers are competing for CPU | Lower `--workers` |

Options can be combined, e.g. `python train.py --workers 12 --epochs 20 --batch-size 256`. Run `python train.py --help` to see them all.

Each epoch prints train loss/accuracy, test accuracy, epoch time, images/sec and peak GPU memory. The checkpoint and `metrics.json` are written to `/workspace/checkpoints/`.

## Task 3: Verify the saved model

On the pod:
```bash
python verify.py 2>&1 | tee task3_verify_log.txt
```
It should print `PASS`, with the reloaded accuracy matching the saved one.

Copy the results back to your laptop:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 -r "root@<POD_IP>:/workspace/checkpoints" "root@<POD_IP>:/workspace/task*.txt" .
```

Optionally re-run the check on your laptop (CPU, needs `pip install torch torchvision pillow`):

```bash
# Quick: 20 bundled test images (2 per class) in sample-data/, no download
python verify.py --ckpt checkpoints/resnet18_cifar10.pt --data-dir ./sample-data

# Full: downloads the CIFAR-10 test set (~170 MB) to ./data and checks it matches the saved accuracy
python verify.py --ckpt checkpoints/resnet18_cifar10.pt --data-dir ./data
```

With `sample-data/`, 20 images are too few to match the saved accuracy exactly, so the check passes at 70% or more. A model that didn't load correctly scores about 10%.

**Stop or terminate the pod when you're done**, because it keeps billing while it's running.
