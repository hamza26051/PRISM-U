"""Fast offline test of the model/training plumbing using synthetic CIFAR-shaped data.
This does not validate CIFAR-10 accuracy; it only checks that forward/backward/checkpoint code can run.
"""
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.resnet import build_resnet18_cifar


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    model = build_resnet18_cifar(10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        break

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test.pt"
        torch.save({"model_state": model.state_dict(), "num_classes": 10}, p)
        ckpt = torch.load(p, map_location=device, weights_only=False)
        model2 = build_resnet18_cifar(10).to(device)
        model2.load_state_dict(ckpt["model_state"])

    print("SMOKE TEST PASSED")
    print(f"device={device}, one-step loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
