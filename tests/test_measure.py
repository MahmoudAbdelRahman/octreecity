import numpy as np

from octreecity.columns import ColumnCity
from octreecity.measure import growth, table
from octreecity.rules import mixed
from octreecity.sources import ColumnSource


def test_growth_rows():
    rng = np.random.default_rng(0)
    h = np.where(rng.random((32, 32)) < 0.3, rng.integers(1, 12, (32, 32)), 0).astype(float)
    src = ColumnSource(ColumnCity.from_heightfield(np.zeros_like(h), h, 2, 16))
    rows = growth(src, mixed(), 3, [2, 1, 0])
    assert [r["min_cell_m"] for r in rows] == [4.0, 2.0, 1.0]
    assert [r["uniform_cells"] for r in rows] == [32 * 32 * 16 // 64, 32 * 32 * 16 // 8, 32 * 32 * 16]
    assert all(r["ratio"] >= 1 for r in rows) and rows[-1]["cells"] > rows[0]["cells"]
    assert np.isfinite(rows[0]["uniform_s"])
    layer = growth(src, mixed(), 3, [1, 0], region=((0, 0, 0), (32, 32, 1)))
    assert [r["uniform_cells"] for r in layer] == [16 * 16, 32 * 32]
    assert all(r["cells"] <= r["uniform_cells"] for r in layer)
    assert "uniform/cells" in table(rows)
