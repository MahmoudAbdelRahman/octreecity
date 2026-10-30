import numpy as np

from octreecity.cli import main
from octreecity.occupancy import Occupancy


def test_info_render_svf_with_region(tmp_path, capsys):
    rng = np.random.default_rng(0)
    h = np.where(rng.random((40, 40)) < 0.2, rng.integers(3, 20, (40, 40)), 0).astype(float)
    city = Occupancy.from_heightfield(np.zeros_like(h) + 1, h + 1, 2, 30)
    path = tmp_path / "c.npz"
    city.save(path)
    main(["info", str(path)])
    assert "columns (40, 40, 30)" in capsys.readouterr().out
    main(["svf", str(path), "--region", "8", "8", "24", "24", "--every", "2", "-o", str(tmp_path / "svf.png")])
    out = capsys.readouterr().out
    n = int(out.split(" points")[0].replace(",", ""))
    assert 0 < n <= 8 * 8                         # only the 16 x 16 crop, every 2nd column
    main(["render", str(path), "-o", str(tmp_path / "top.png")])
    assert (tmp_path / "top.png").exists()
