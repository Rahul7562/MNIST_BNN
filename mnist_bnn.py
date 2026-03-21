"""
train_bnn.py  –  High-Accuracy BNN for MNIST (target: 94-96%)
================================================================================
Architecture : 784 → 512 (BNN) → 256 (BNN) → 10 (softmax)
Output       : argmax of 10 classes → converted to 4-bit for Verilog
               digit 2 → argmax=2 → 4'b0010

Key fixes over the broken 60%-stuck version:
  [FIX-A] Output is now 10 classes + CrossEntropyLoss, NOT 4 bits + BCELoss.
          The 4-bit binary encoding is applied ONLY at export time for Verilog.
          Original code forced the network to learn digit 6=0110 vs 7=0111 differ
          by 1 bit even though they look nothing alike — an impossible task.

  [FIX-B] Hidden layers: 512 + 256 instead of just 64.
          BNNs lose information from binarisation; they need ~8x more neurons
          than a full-precision network to compensate.

  [FIX-C] Two hidden layers instead of one.
          More depth = richer features = better generalisation.

  [FIX-D] BatchNorm with affine=True (learnable gamma, beta) instead of False.
          Allows each layer to find its own best activation scale/shift.

  [FIX-E] Cosine annealing LR schedule (decays LR smoothly to near-zero).
          Fixed LR causes oscillation in later epochs — the model overshoots
          the minimum and accuracy flatlines at ~60%.

  [FIX-F] Input normalisation (mean=0.1307, std=0.3081) — standard for MNIST.
          Without it pixels are in [0,1] which is a suboptimal STE input.

  [FIX-G] Binarise pixel by sign(normalised_pixel) not sign(2x-1).
          Normalised inputs have a better zero-crossing for binarisation.

All previously-fixed bugs are retained:
  [BUG1] typing.Dict for Python 3.7/3.8 compat
  [BUG2] __main__ guard for DataLoader multiprocessing
  [BUG3] .to(device) on targets
  [BUG4] .squeeze(0) not .squeeze()

Usage:
    pip install torch torchvision
    python train_bnn.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
N_INPUT   = 784
N_H1      = 512          # FIX-B: was 64, needs to be much larger
N_H2      = 256          # FIX-C: second hidden layer
N_CLASSES = 10           # FIX-A: 10 class outputs, not 4 bits
N_OUT     = 4            # Verilog output width (kept for export only)
EPOCHS    = 40
LR        = 1e-3
BATCH     = 256
SEED      = 42
OUT_DIR   = "mem_files"

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

# ── BNN Layer helper ──────────────────────────────────────────────────────────
class BNNLinear(nn.Module):
    """
    One BNN hidden layer:
      binarise(input) → binary matmul → BatchNorm → binarise(output)
    BatchNorm has affine=True (FIX-D) so it can learn its own scale/shift.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bn     = nn.BatchNorm1d(out_features, affine=True)  # FIX-D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wb  = bsign(self.weight)          # binarise weights {-1, +1}
        xb  = bsign(x)                    # binarise input   {-1, +1}
        out = xb @ wb.t()                 # binary dot product (integer result)
        out = self.bn(out)                # normalise
        return out                        # caller decides whether to binarise

# ── BNN Model ─────────────────────────────────────────────────────────────────
class BNN(nn.Module):
    """
    784 → [BNNLinear(512) → sign] → [BNNLinear(256) → sign] → Linear(10)
    Last layer is full-precision so CrossEntropyLoss has proper logits to work with.
    """
    def __init__(self):
        super().__init__()
        self.l1 = BNNLinear(N_INPUT, N_H1)      # FIX-B, FIX-C, FIX-D
        self.l2 = BNNLinear(N_H1,   N_H2)       # FIX-C
        # Final layer: real-valued weights, no binarisation → proper softmax logits
        self.fc_out = nn.Linear(N_H2, N_CLASSES) # FIX-A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = x.view(-1, N_INPUT)
        # FIX-G: sign of already-normalised pixel is a better binariser than 2x-1
        h1 = bsign(self.l1(bsign(x)))
        h2 = bsign(self.l2(h1))
        return self.fc_out(h2)              # shape: (batch, 10) — raw logits


if __name__ == '__main__':

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Dataset with normalisation (FIX-F) ───────────────────────────────────
    # Standard MNIST mean/std: computed over all 60K training pixels
    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # FIX-F
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # FIX-F
    ])

    train_ds     = datasets.MNIST("./data", train=True,  download=True, transform=tfm_train)
    test_ds      = datasets.MNIST("./data", train=False, download=True, transform=tfm_test)
    # Use raw pixels (no normalisation) for saving image .mem files
    raw_test_ds  = datasets.MNIST("./data", train=False, download=False,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    # ── Model, optimiser, loss ────────────────────────────────────────────────
    model     = BNN().to(device)
    optim_    = optim.Adam(model.parameters(), lr=LR)

    # FIX-E: Cosine annealing — LR decays smoothly from LR to ~0 over EPOCHS
    # This prevents the late-epoch oscillation that flatlines accuracy at 60%.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=EPOCHS, eta_min=1e-5)

    # FIX-A: CrossEntropyLoss on 10 classes — learns real visual structure
    criterion = nn.CrossEntropyLoss()

    print(f"Training BNN  [{N_INPUT} → {N_H1} → {N_H2} → {N_CLASSES}]  for {EPOCHS} epochs …\n")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>7}  {'LR':>10}")
    print("  " + "─" * 36)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)       # long tensor, no conversion needed

            optim_.zero_grad()
            logits = model(imgs)             # (batch, 10)
            loss   = criterion(logits, labels)
            loss.backward()
            optim_.step()
            total_loss += loss.item()

        scheduler.step()                     # FIX-E: update LR after each epoch

        # ── Eval ──────────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                preds  = model(imgs).argmax(dim=1)   # class with highest logit
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc = correct / total * 100
        cur_lr = scheduler.get_last_lr()[0]
        print(f"  {epoch:>5}  {total_loss/len(train_loader):>8.4f}  {acc:>6.2f}%  {cur_lr:>10.6f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "bnn_best.pt")

    print(f"\n  Best test accuracy: {best_acc:.2f}%")
    print("\nTraining complete. Loading best weights for export …\n")

    # Load the best checkpoint before exporting
    model.load_state_dict(torch.load("bnn_best.pt", map_location=device))
    model.eval()

    # ─────────────────────────────────────────────────────────────────────────
    #  Export weights to .mem files
    #  The 4-bit binary encoding is ONLY used here for the Verilog design.
    #  The network itself outputs 10 logits; argmax gives the digit.
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    #  Export weights to .mem files
    # ─────────────────────────────────────────────────────────────────────────
    print("Exporting weights …")

    # ✅ Ensure directory exists (fix)
    os.makedirs(OUT_DIR, exist_ok=True)

    with torch.no_grad():
        w1_bin = ((bsign(model.l1.weight) + 1) // 2).int().cpu().numpy()

        gamma1  = model.l1.bn.weight.cpu().numpy()
        beta1   = model.l1.bn.bias.cpu().numpy()
        mean1   = model.l1.bn.running_mean.cpu().numpy()
        var1    = model.l1.bn.running_var.cpu().numpy()
        std1    = np.sqrt(var1 + 1e-5)
        adj_mu1 = mean1 - beta1 * std1 / (gamma1 + 1e-8)
        thresh1 = np.clip(np.round((N_INPUT + adj_mu1) / 2).astype(int), 0, N_INPUT)

        w2_bin  = ((bsign(model.l2.weight) + 1) // 2).int().cpu().numpy()

        gamma2  = model.l2.bn.weight.cpu().numpy()
        beta2   = model.l2.bn.bias.cpu().numpy()
        mean2   = model.l2.bn.running_mean.cpu().numpy()
        var2    = model.l2.bn.running_var.cpu().numpy()
        std2    = np.sqrt(var2 + 1e-5)
        adj_mu2 = mean2 - beta2 * std2 / (gamma2 + 1e-8)
        thresh2 = np.clip(np.round((N_H1 + adj_mu2) / 2).astype(int), 0, N_H1)

        w_out   = model.fc_out.weight.cpu().numpy()
        b_out   = model.fc_out.bias.cpu().numpy()

    # ── Write files (FIXED PATHS) ─────────────────────────────────────────────

    with open(os.path.join(OUT_DIR, "weights_l1.mem"), "w") as f:
        for row in w1_bin:
            f.write("".join(map(str, row)) + "\n")

    with open(os.path.join(OUT_DIR, "thresh_l1.mem"), "w") as f:
        for t in thresh1:
            f.write(format(int(np.clip(t, 0, 1023)), "010b") + "\n")

    with open(os.path.join(OUT_DIR, "weights_l2.mem"), "w") as f:
        for row in w2_bin:
            f.write("".join(map(str, row)) + "\n")

    with open(os.path.join(OUT_DIR, "thresh_l2.mem"), "w") as f:
        for t in thresh2:
            f.write(format(int(np.clip(t, 0, 1023)), "010b") + "\n")

    SCALE = 256
    w_out_fx = np.clip(np.round(w_out * SCALE).astype(int), -32768, 32767)
    b_out_fx = np.clip(np.round(b_out * SCALE).astype(int), -32768, 32767)

    with open(os.path.join(OUT_DIR, "weights_out.mem"), "w") as f:
        for row in w_out_fx:
            f.write(" ".join(format(int(v) & 0xFFFF, "04X") for v in row) + "\n")

    with open(os.path.join(OUT_DIR, "bias_out.mem"), "w") as f:
        f.write(" ".join(format(int(v) & 0xFFFF, "04X") for v in b_out_fx) + "\n")

    print(f"  ✓  files saved in {OUT_DIR}/")

    # ── Export one raw test image per digit + verify predictions ─────────────
    print("\nExporting test images and verifying …\n")

    found: Dict[int, torch.Tensor] = {}
    for img, label in raw_test_ds:          # raw pixels [0,1], no normalisation
        d = int(label)
        if d not in found:
            found[d] = img
        if len(found) == 10:
            break

    # Normalise separately for model inference
    norm = transforms.Normalize((0.1307,), (0.3081,))

    print(f"  {'Digit':>5}  {'Expected bits':>13}  {'Got bits':>8}  {'Pred':>4}  Status")
    print("  " + "─" * 48)

    with torch.no_grad():
        for digit in range(10):
            raw_img  = found[digit]                         # (1, 28, 28) in [0,1]
            norm_img = norm(raw_img).to(device)

            # Run inference
            logits   = model(norm_img.unsqueeze(0))         # (1, 10)
            pred     = int(logits.argmax(dim=1).item())     # predicted digit

            # Convert to 4-bit binary (FIX-A: done here, not during training)
            exp_bits = format(digit, "04b")
            got_bits = format(pred,  "04b")
            status   = "✓ CORRECT" if pred == digit else "✗ WRONG"

            print(f"  {digit:>5}  {exp_bits:>13}  {got_bits:>8}  {pred:>4}  {status}")

            # Save binarised pixel image for Verilog testbench
            img_bin = (raw_img.view(-1) > 0.5).int().numpy()
            with open(f"{OUT_DIR}/test_image_{digit}.mem", "w") as f:
                f.write("".join(map(str, img_bin)) + "\n")

    print(f"\n  All test images saved to {OUT_DIR}/test_image_N.mem")
    print("\n" + "=" * 52)
    print(f"  DONE.  Best accuracy: {best_acc:.2f}%")
    print( "  Load mem_files/ into your Verilog design.")
    print("=" * 52)

    # ── Summary of Verilog changes needed ─────────────────────────────────────
    print("""
NOTE — update bnn_top.v for the new architecture:
  • Layer 1: N_HIDDEN1 = 512,  width of weight ROM = 784
  • Layer 2: N_HIDDEN2 = 256,  width of weight ROM = 512
  • Output:  10 accumulator units (Q8.8 MAC), argmax → 4-bit digit_out
  • State machine: LAYER1 (512 cycles) → LAYER2 (256 cycles) → OUTPUT (10 cycles)
  Total latency: 512 + 256 + 10 + 2 = 780 cycles @ 100 MHz = 7.8 µs
""")