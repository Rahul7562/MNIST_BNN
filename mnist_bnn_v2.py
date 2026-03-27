"""
mnist_bnn_v2.py - High-Accuracy BNN for MNIST with FPGA Verification
================================================================================
Architecture : 784 -> 512 (BNN) -> 256 (BNN) -> 10 (softmax)
Target       : Zynq-7000 FPGA with switch-based image selection

Key improvements:
  1. Robust threshold calculation with proper handling of BatchNorm signs
  2. Hardware-equivalent inference verification in Python
  3. Export multiple test images per digit for switch selection
  4. Detailed diagnostics for debugging hardware mismatches

Usage:
    pip install torch torchvision numpy
    python mnist_bnn_v2.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
N_INPUT   = 784
N_H1      = 512
N_H2      = 256
N_CLASSES = 10
EPOCHS    = 50      # More epochs for better convergence
LR        = 1e-3
BATCH     = 256
SEED      = 42
OUT_DIR   = "mem_files"
NUM_TEST_IMAGES_PER_DIGIT = 4  # Export 4 images per digit for switch selection

# MNIST normalization
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

# ── Straight-Through Estimator ────────────────────────────────────────────────
class BinarySignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        g = grad.clone()
        g[x.abs() > 1] = 0
        return g

bsign = BinarySignSTE.apply

# ── BNN Layer ─────────────────────────────────────────────────────────────────
class BNNLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bn = nn.BatchNorm1d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wb = bsign(self.weight)
        xb = bsign(x)
        out = xb @ wb.t()
        out = self.bn(out)
        return out

# ── BNN Model ─────────────────────────────────────────────────────────────────
class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BNNLinear(N_INPUT, N_H1)
        self.l2 = BNNLinear(N_H1, N_H2)
        self.fc_out = nn.Linear(N_H2, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, N_INPUT)
        h1 = bsign(self.l1(bsign(x)))
        h2 = bsign(self.l2(h1))
        return self.fc_out(h2)


# ── Hardware-Equivalent Inference ─────────────────────────────────────────────
def hw_inference(image_bin: np.ndarray,
                 w1_bin: np.ndarray, thresh1: np.ndarray, invert1: np.ndarray,
                 w2_bin: np.ndarray, thresh2: np.ndarray, invert2: np.ndarray,
                 w_out: np.ndarray, b_out: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Simulate exact hardware behavior for verification.
    image_bin: 784 bits (0/1)
    Returns: predicted digit, output scores array
    """
    # Layer 1: 512 neurons
    hidden1 = np.zeros(N_H1, dtype=np.int32)
    for i in range(N_H1):
        # XNOR = count matches = popcount(~(input ^ weight))
        xnor = ~(image_bin ^ w1_bin[i]) & 1  # bit-wise XNOR
        popcount = np.sum(xnor)
        # Compare with threshold, XOR with invert flag
        hidden1[i] = (int(popcount >= thresh1[i]) ^ invert1[i]) & 1

    # Layer 2: 256 neurons
    hidden2 = np.zeros(N_H2, dtype=np.int32)
    for i in range(N_H2):
        xnor = ~(hidden1 ^ w2_bin[i]) & 1
        popcount = np.sum(xnor)
        hidden2[i] = (int(popcount >= thresh2[i]) ^ invert2[i]) & 1

    # Output layer: MAC with signed weights
    scores = np.zeros(N_CLASSES, dtype=np.int64)
    for c in range(N_CLASSES):
        acc = int(b_out[c])  # Start with bias
        for j in range(N_H2):
            w = int(w_out[c, j])
            # hidden bit: 1 means +1, 0 means -1
            if hidden2[j]:
                acc += w
            else:
                acc -= w
        scores[c] = acc

    return int(np.argmax(scores)), scores


def compute_thresholds(model: BNN, device: torch.device):
    """
    Compute thresholds and invert flags from BatchNorm parameters.

    For BNN layer: popcount(XNOR(input, weight)) >= threshold

    The BatchNorm transforms: y = gamma * (x - mean) / std + beta
    For binary output (sign(y) > 0), we need: gamma * (x - mean) / std + beta > 0

    If gamma > 0: x > mean - beta * std / gamma
    If gamma < 0: x < mean - beta * std / gamma  (inequality flips!)

    For popcount-based computation:
    x = 2 * popcount - N  (converting from popcount to bipolar sum)
    threshold = (N + adj_mu) / 2, where adj_mu = mean - beta * std / gamma
    """
    model.eval()
    with torch.no_grad():
        # Layer 1
        w1_float = model.l1.weight.cpu().numpy()
        w1_bin = ((np.sign(w1_float) + 1) // 2).astype(np.int32)

        gamma1 = model.l1.bn.weight.cpu().numpy()
        beta1 = model.l1.bn.bias.cpu().numpy()
        mean1 = model.l1.bn.running_mean.cpu().numpy()
        var1 = model.l1.bn.running_var.cpu().numpy()
        std1 = np.sqrt(var1 + 1e-5)

        # Compute adjusted mean for threshold
        # Use abs(gamma) to avoid division issues, handle sign with invert flag
        gamma1_abs = np.maximum(np.abs(gamma1), 1e-8)
        adj_mu1 = mean1 - beta1 * std1 / gamma1_abs * np.sign(gamma1)
        thresh1 = np.clip(np.round((N_INPUT + adj_mu1) / 2).astype(int), 0, N_INPUT)
        invert1 = (gamma1 < 0).astype(np.int32)

        # Layer 2
        w2_float = model.l2.weight.cpu().numpy()
        w2_bin = ((np.sign(w2_float) + 1) // 2).astype(np.int32)

        gamma2 = model.l2.bn.weight.cpu().numpy()
        beta2 = model.l2.bn.bias.cpu().numpy()
        mean2 = model.l2.bn.running_mean.cpu().numpy()
        var2 = model.l2.bn.running_var.cpu().numpy()
        std2 = np.sqrt(var2 + 1e-5)

        gamma2_abs = np.maximum(np.abs(gamma2), 1e-8)
        adj_mu2 = mean2 - beta2 * std2 / gamma2_abs * np.sign(gamma2)
        thresh2 = np.clip(np.round((N_H1 + adj_mu2) / 2).astype(int), 0, N_H1)
        invert2 = (gamma2 < 0).astype(np.int32)

        # Output layer (Q8.8 fixed point, scale by 256)
        SCALE = 256
        w_out = model.fc_out.weight.cpu().numpy()
        b_out = model.fc_out.bias.cpu().numpy()
        w_out_fx = np.clip(np.round(w_out * SCALE).astype(np.int32), -32768, 32767)
        b_out_fx = np.clip(np.round(b_out * SCALE).astype(np.int32), -32768, 32767)

    return {
        'w1_bin': w1_bin, 'thresh1': thresh1, 'invert1': invert1,
        'w2_bin': w2_bin, 'thresh2': thresh2, 'invert2': invert2,
        'w_out_fx': w_out_fx, 'b_out_fx': b_out_fx
    }


def verify_model(model: BNN, test_ds, device: torch.device, params: dict) -> float:
    """
    Run hardware-equivalent inference on test set and compare to model output.
    """
    model.eval()
    norm = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

    hw_correct = 0
    sw_correct = 0
    total = 0
    mismatches = 0

    print("\n" + "="*60)
    print("Hardware-Software Verification")
    print("="*60)

    with torch.no_grad():
        for idx, (raw_img, label) in enumerate(test_ds):
            if idx >= 1000:  # Test on first 1000 images
                break

            # Software inference
            norm_img = norm(raw_img).to(device)
            logits = model(norm_img.unsqueeze(0))
            sw_pred = int(logits.argmax(dim=1).item())

            # Convert to hardware format
            raw_flat = raw_img.view(-1).cpu().numpy()
            norm_flat = (raw_flat - MNIST_MEAN) / MNIST_STD
            img_bin = (norm_flat >= 0).astype(np.int32)

            # Hardware inference
            hw_pred, _ = hw_inference(
                img_bin,
                params['w1_bin'], params['thresh1'], params['invert1'],
                params['w2_bin'], params['thresh2'], params['invert2'],
                params['w_out_fx'], params['b_out_fx']
            )

            label_int = int(label)
            if sw_pred == label_int:
                sw_correct += 1
            if hw_pred == label_int:
                hw_correct += 1
            if sw_pred != hw_pred:
                mismatches += 1
            total += 1

    hw_acc = hw_correct / total * 100
    sw_acc = sw_correct / total * 100

    print(f"  Software accuracy: {sw_acc:.2f}% ({sw_correct}/{total})")
    print(f"  Hardware accuracy: {hw_acc:.2f}% ({hw_correct}/{total})")
    print(f"  HW/SW mismatches:  {mismatches}/{total} ({mismatches/total*100:.2f}%)")

    if mismatches > total * 0.05:
        print("\n  WARNING: High mismatch rate! Check threshold calculation.")

    return hw_acc


def export_weights(params: dict, out_dir: str):
    """Export all weights and parameters to .mem files for Vivado.

    IMPORTANT: Binary weights are written in REVERSED order because Verilog's
    $readmemb reads the first character as MSB (bit N-1), not LSB (bit 0).
    This ensures bit alignment between Python (index 0 = first element) and
    Verilog (bit 0 = LSB, bit N-1 = MSB read from first char).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Layer 1 binary weights (512 x 784 bits) - REVERSED for Verilog
    with open(os.path.join(out_dir, "weights_l1.mem"), "w") as f:
        for row in params['w1_bin']:
            f.write("".join(map(str, row[::-1])) + "\n")  # Reverse bit order

    # Layer 1 thresholds (10-bit binary)
    with open(os.path.join(out_dir, "thresh_l1.mem"), "w") as f:
        for t in params['thresh1']:
            f.write(format(int(np.clip(t, 0, 1023)), "010b") + "\n")

    # Layer 1 invert flags
    with open(os.path.join(out_dir, "invert_l1.mem"), "w") as f:
        for inv in params['invert1']:
            f.write(f"{inv}\n")

    # Layer 2 binary weights (256 x 512 bits) - REVERSED for Verilog
    with open(os.path.join(out_dir, "weights_l2.mem"), "w") as f:
        for row in params['w2_bin']:
            f.write("".join(map(str, row[::-1])) + "\n")  # Reverse bit order

    # Layer 2 thresholds (10-bit binary)
    with open(os.path.join(out_dir, "thresh_l2.mem"), "w") as f:
        for t in params['thresh2']:
            f.write(format(int(np.clip(t, 0, 1023)), "010b") + "\n")

    # Layer 2 invert flags
    with open(os.path.join(out_dir, "invert_l2.mem"), "w") as f:
        for inv in params['invert2']:
            f.write(f"{inv}\n")

    # Output layer weights (10 rows, 256 hex values each) - NO reversal for $readmemh
    # Unlike binary weights, hex values are read sequentially into flat array
    with open(os.path.join(out_dir, "weights_out.mem"), "w") as f:
        for row in params['w_out_fx']:
            f.write(" ".join(format(int(v) & 0xFFFF, "04X") for v in row) + "\n")

    # Output layer bias (10 hex values) - no reversal needed (not bit-packed)
    with open(os.path.join(out_dir, "bias_out.mem"), "w") as f:
        f.write(" ".join(format(int(v) & 0xFFFF, "04X") for v in params['b_out_fx']) + "\n")

    print(f"\n  Weights exported to {out_dir}/")


def export_test_images(test_ds, params: dict, out_dir: str, images_per_digit: int = 4):
    """Export multiple test images per digit for switch selection."""
    os.makedirs(out_dir, exist_ok=True)

    # Collect images per digit
    digit_images: Dict[int, List[Tuple[np.ndarray, int]]] = {d: [] for d in range(10)}

    norm = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

    for raw_img, label in test_ds:
        d = int(label)
        if len(digit_images[d]) < images_per_digit:
            raw_flat = raw_img.view(-1).cpu().numpy()
            norm_flat = (raw_flat - MNIST_MEAN) / MNIST_STD
            img_bin = (norm_flat >= 0).astype(np.int32)

            # Verify with hardware inference
            hw_pred, scores = hw_inference(
                img_bin,
                params['w1_bin'], params['thresh1'], params['invert1'],
                params['w2_bin'], params['thresh2'], params['invert2'],
                params['w_out_fx'], params['b_out_fx']
            )

            # Store image, actual label, and prediction
            digit_images[d].append((img_bin, hw_pred, scores))

        # Check if we have enough
        if all(len(imgs) >= images_per_digit for imgs in digit_images.values()):
            break

    print("\n" + "="*60)
    print("Test Image Export with Hardware Verification")
    print("="*60)
    print(f"  {'Digit':<6} {'Img#':<5} {'HW Pred':<8} {'Status':<10} {'Confidence'}")
    print("  " + "-"*50)

    # Export images - REVERSED for Verilog $readmemb (first char = MSB)
    for digit in range(10):
        for img_idx, (img_bin, hw_pred, scores) in enumerate(digit_images[digit]):
            # Save image with REVERSED bit order for Verilog
            filename = f"test_image_{digit}_{img_idx}.mem"
            with open(os.path.join(out_dir, filename), "w") as f:
                f.write("".join(map(str, img_bin.tolist()[::-1])) + "\n")

            # Calculate confidence (margin between top 2 scores)
            sorted_scores = np.sort(scores)[::-1]
            confidence = sorted_scores[0] - sorted_scores[1]

            status = "CORRECT" if hw_pred == digit else "WRONG"
            print(f"  {digit:<6} {img_idx:<5} {hw_pred:<8} {status:<10} {confidence}")

    # Also export single test images (for backwards compatibility) - REVERSED
    for digit in range(10):
        if digit_images[digit]:
            img_bin, _, _ = digit_images[digit][0]
            with open(os.path.join(out_dir, f"test_image_{digit}.mem"), "w") as f:
                f.write("".join(map(str, img_bin.tolist()[::-1])) + "\n")

    print(f"\n  {images_per_digit * 10} test images exported to {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Dataset
    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm_train)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm_test)
    raw_test_ds = datasets.MNIST("./data", train=False, download=False,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Model
    model = BNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    print(f"Training BNN [{N_INPUT} -> {N_H1} -> {N_H2} -> {N_CLASSES}] for {EPOCHS} epochs...\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>7}  {'LR':>10}")
    print("  " + "-" * 36)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  {epoch:>5}  {total_loss/len(train_loader):>8.4f}  {acc:>6.2f}%  {cur_lr:>10.6f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "bnn_best.pt")

    print(f"\n  Best test accuracy: {best_acc:.2f}%")
    print("\nLoading best weights for export...")

    model.load_state_dict(torch.load("bnn_best.pt", map_location=device, weights_only=True))
    model.eval()

    # Compute parameters
    params = compute_thresholds(model, device)

    # Verify hardware-software equivalence
    hw_acc = verify_model(model, raw_test_ds, device, params)

    # Export weights
    export_weights(params, OUT_DIR)

    # Export test images (4 per digit = 40 total for switch selection)
    export_test_images(raw_test_ds, params, OUT_DIR, NUM_TEST_IMAGES_PER_DIGIT)

    # Summary
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"  Software accuracy: {best_acc:.2f}%")
    print(f"  Hardware accuracy: {hw_acc:.2f}%")
    print(f"  Files exported to: {OUT_DIR}/")
    print("\nNext steps:")
    print("  1. Run sync_mem_files.bat to copy to Vivado simulation dir")
    print("  2. Run simulation in Vivado to verify")
    print("  3. Synthesize and implement for Zynq-7000")
    print("="*60)
