import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from opacus import PrivacyEngine

device = "cuda" if torch.cuda.is_available() else "cpu"

ds = datasets.MNIST("data", train=True, download=True)
perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:5000]
data = ds.data[perm].float().div(255).sub(0.1307).div(0.3081).unsqueeze(1)
labels = ds.targets[perm]
zx, zy = data[0].to(device), labels[0].to(device)   # the special example z
X, y = data[1:], labels[1:]                          # the other 4,999

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))

def train_once(include, sigma, seed, epochs=6, lr=0.5):
    torch.manual_seed(seed)
    Xt = torch.cat([X, zx.cpu()[None]]) if include else X
    yt = torch.cat([y, zy.cpu()[None]]) if include else y
    model = Net().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=100)
    model, opt, loader = PrivacyEngine().make_private(
        module=model, optimizer=opt, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=1.0)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    with torch.no_grad():
        return -F.cross_entropy(model(zx[None]), zy[None]).item()   # score for W