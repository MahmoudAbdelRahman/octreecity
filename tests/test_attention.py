import numpy as np
import pytest

torch = pytest.importorskip("torch")

from octreecity.attention import LinkModel, fit, predict  # noqa: E402
from octreecity.build import build  # noqa: E402
from octreecity.columns import ColumnCity  # noqa: E402
from octreecity.learn import tokens  # noqa: E402
from octreecity.links import hierarchy, neighbours  # noqa: E402
from octreecity.rules import mixed  # noqa: E402
from octreecity.sources import ColumnSource  # noqa: E402


def tile(seed):
    rng = np.random.default_rng(seed)
    h = np.where(rng.random((32, 32)) < 0.3, rng.integers(1, 12, (32, 32)), 0).astype(float)
    t = build(ColumnSource(ColumnCity.from_heightfield(np.zeros_like(h), h, 2, 16)), mixed(), 3).balance(6)
    t.cells["built"] = (t.label == 2).astype(float)
    sel = t.in_box((0, 0, 0), (32, 32, 1))
    # target: share of built cells nearby (a smooth function of the geometry)
    t.cells["y"] = np.convolve(t.cells["built"], np.ones(5) / 5, "same")
    p = t.parents(levels_above=1)
    return tokens(t, ["built", "size_m"], neighbours(t, mask=sel) | hierarchy(t, p), targets=["y"],
                  select=sel, parents=p, k=16)


def test_model_shapes_and_training_lowers_loss():
    sets = [tile(s) for s in range(4)]
    model = LinkModel.from_tokens(sets[0], dim=32, heads=4, layers=2)
    assert model.layers[0].kind.num_embeddings == len(sets[0].kinds) + 1
    hist = fit(model, sets, steps=60, device="cpu")
    assert np.mean(hist[-10:]) < np.mean(hist[:10])
    out = predict(model, sets[0])
    assert out.shape == (len(sets[0]), 1) and np.isfinite(out).all()


def test_checkpointing_gives_same_output():
    tok = tile(0)
    m = LinkModel.from_tokens(tok, dim=16, heads=2, layers=2, checkpoint_above=10).train()
    t = tok.to_torch()
    a = m.run(t)
    m.checkpoint_above = None
    assert torch.allclose(a, m.run(t))
