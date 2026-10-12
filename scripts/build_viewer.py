"""Inline viewer/data_tile<N>.js into viewer/template.html -> viewer/tile<N>.html."""

import sys
from pathlib import Path

V = Path(__file__).resolve().parent.parent / "viewer"
tile = int(sys.argv[1]) if len(sys.argv) > 1 else 868
html = (V / "template.html").read_text().replace("/*DATA*/", (V / f"data_tile{tile}.js").read_text())
(V / f"tile{tile}.html").write_text(html)
print(V / f"tile{tile}.html")
