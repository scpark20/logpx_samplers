# minimal_pytorch_train.py  (<=100 lines)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fake regression data: y = 3x + 2 + noise
    N = 4096
    x = torch.randn(N, 1)
    y = 3.0 * x + 2.0 + 0.1 * torch.randn(N, 1)

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    # Tiny model
    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train
    for epoch in range(5):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)

        avg = total / len(ds)
        with torch.no_grad():
            w0 = model[0].weight.data.mean().item()
            b0 = model[0].bias.data.mean().item()
        print(f"epoch {epoch+1}: loss={avg:.6f} (layer0 mean w={w0:.3f}, b={b0:.3f})")

    # Quick inference
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor([[0.0], [1.0], [2.0]], device=device)
        test_y = model(test_x).cpu().view(-1)
    print("pred at x=0,1,2:", [round(v, 3) for v in test_y.tolist()])

if __name__ == "__main__":
    main()
