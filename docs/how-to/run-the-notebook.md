# Running the notebook

```bash
cd octreecity
./demo.sh
```

`demo.sh` installs everything and opens the quickstart notebook in the
project's own JupyterLab. That JupyterLab includes the plotly plugin, so the 3D
views draw offline, with no "trust" step. In VS Code, open the notebook and
choose the project's `.venv` kernel.

## If a 3D view stays blank

The 3D views are drawn by a JupyterLab plugin that comes with plotly. The
JupyterLab you use must come from the same environment as plotly:

- Start Jupyter from the project (`./demo.sh` or `uv run jupyter lab`), not
  with `uv run --with jupyterlab ...`. That starts JupyterLab from a separate
  environment without the plugin.
- Check with `uv run jupyter labextension list`. It should list
  `jupyterlab-plotly ... enabled OK`.
- The notebook's first cell calls `setup_notebook()`. It also sends each view
  as a web page, so other front ends (classic Jupyter, nbviewer) still show it
  when online.

## Before a presentation

- Run all cells once (Kernel > Restart and Run All, about 40 s). The
  OpenStreetMap download is then cached in `data/`, and the notebook works
  without internet.
- As a backup, a web-page version of the notebook can be built with
  `uv run python scripts/notebook_report.py` (writes
  `outputs/report/quickstart_page.html`).
