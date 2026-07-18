"""Commit-2 correctness (spec): identical seeds -> identical 50-step loss
sequences; different seeds -> different. CPU exact; TinyNet stands in for the
real model so this runs without data/GPU."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from audioshield.utils.seeding import seed_everything, dataloader_seed_kwargs, make_generator

def _micro_run(seed: int, steps: int = 50):
    rec = seed_everything(seed)
    assert rec["seed"] == seed
    X = torch.randn(256, 16)          # created AFTER seeding -> part of the determinism test
    y = (X.sum(dim=1) > 0).float()
    dl = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True,
                    **dataloader_seed_kwargs(seed))
    torch.manual_seed(seed)           # model init
    net = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    losses, it = [], iter(dl)
    for _ in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl); xb, yb = next(it)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xb).squeeze(-1), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return np.array(losses)

def test_same_seed_identical():
    a, b = _micro_run(13), _micro_run(13)
    assert np.array_equal(a, b), f"non-deterministic: max delta {np.abs(a-b).max()}"

def test_different_seed_differs():
    assert not np.array_equal(_micro_run(13), _micro_run(14))

def test_dataloader_order_reproducible():
    ds = TensorDataset(torch.arange(100).float().unsqueeze(1))
    o1 = [int(b[0][0, 0]) for b in DataLoader(ds, batch_size=10, shuffle=True, generator=make_generator(13, 1))]
    o2 = [int(b[0][0, 0]) for b in DataLoader(ds, batch_size=10, shuffle=True, generator=make_generator(13, 1))]
    o3 = [int(b[0][0, 0]) for b in DataLoader(ds, batch_size=10, shuffle=True, generator=make_generator(14, 1))]
    assert o1 == o2 and o1 != o3

def test_non_int_seed_rejected():
    import pytest
    with pytest.raises(TypeError):
        seed_everything("13")   # catches the classic unplumbed-config-string bug
