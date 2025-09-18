
# nn_reducer.py
# Autoencoder 3D para reducción no lineal del embedding ideológico.
# Compatible con CPU o GPU. Guarda el modelo y produce coordenadas 3D.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    _TORCH_ERR = repr(e)


@dataclass
class AEConfig:
    input_dim: int
    hidden1: int = 128
    hidden2: int = 64
    bottleneck: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 1500
    patience: int = 50
    device: str = "cuda"


class MLP_AE(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        D = cfg.input_dim
        H1, H2, B = cfg.hidden1, cfg.hidden2, cfg.bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(D, H1),
            nn.ReLU(),
            nn.BatchNorm1d(H1),
            nn.Dropout(cfg.dropout),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.BatchNorm1d(H2),
            nn.Linear(H2, B),
        )
        self.decoder = nn.Sequential(
            nn.Linear(B, H2),
            nn.ReLU(),
            nn.BatchNorm1d(H2),
            nn.Linear(H2, H1),
            nn.ReLU(),
            nn.BatchNorm1d(H1),
            nn.Dropout(cfg.dropout),
            nn.Linear(H1, D),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(X: np.ndarray, cfg: AEConfig, out_dir: Path) -> Tuple[np.ndarray, dict]:
    if not TORCH_OK:
        raise RuntimeError(f"PyTorch no disponible: {_TORCH_ERR}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Estandarización por dimensión
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma

    # Split train/val
    n = Xn.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(n * 0.85)
    tr_idx, va_idx = idx[:split], idx[split:]
    Xtr = torch.tensor(Xn[tr_idx], dtype=torch.float32)
    Xva = torch.tensor(Xn[va_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(Xva), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = MLP_AE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    bad = 0
    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            xhat, _ = model(xb)
            loss = loss_fn(xhat, xb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xhat, _ = model(xb)
                loss = loss_fn(xhat, xb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= max(1, len(val_loader.dataset))

        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), out_dir / "ae3d_model.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    # Cargar el mejor y proyectar todo
    model.load_state_dict(torch.load(out_dir / "ae3d_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        Xten = torch.tensor(Xn, dtype=torch.float32).to(device)
        _, Z = model(Xten)
        Z = Z.cpu().numpy()

    # Guardar stats y config
    meta = {
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "cfg": cfg.__dict__,
        "best_val": best_val,
        "history": history,
        "device_used": device,
    }
    (out_dir / "ae3d_stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Z, meta


def project_with_trained(X: np.ndarray, model_dir: Path) -> np.ndarray:
    if not TORCH_OK:
        raise RuntimeError(f"PyTorch no disponible: {_TORCH_ERR}")
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "ae3d_stats.json").read_text(encoding="utf-8"))
    mu = np.array(meta["mu"])
    sigma = np.array(meta["sigma"])
    cfg = AEConfig(**meta["cfg"])
    device = cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

    Xn = (X - mu) / sigma
    model = MLP_AE(cfg)
    model.load_state_dict(torch.load(model_dir / "ae3d_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        import torch
        Z = model.encoder(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return Z
