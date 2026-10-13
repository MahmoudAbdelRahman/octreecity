"""Run the quickstart notebook and write it as one shareable web page.

Text, results and interactive 3D views (plotly.js from jsDelivr); code sits
in fold-out boxes. Figures are simplified a little more than in Jupyter to
keep the page light.

Usage: .venv/bin/python scripts/notebook_report.py
"""

import html
import json
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks/octreecity_quickstart.ipynb"
OUT = ROOT / "outputs/report/quickstart_page.html"
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"

nb = nbformat.read(SRC, 4)
nb.cells.insert(2, nbformat.v4.new_code_cell("import octreecity.viz as _v; _v.DISPLAY_COLUMNS = 70_000"))
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(SRC.parent)}}).execute()
del nb.cells[2]


def inline(t):
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", t)
    return t


def markdown(src):
    out, para, items = [], [], []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            text = s.lstrip("#").strip()
            if level == 1:
                continue
            m = re.match(r"(\d+)\.\s+(.*)", text)
            if m:
                out.append(f'<h2><span class="step">{m.group(1)}</span>{inline(m.group(2))}</h2>')
            else:
                out.append(f"<h2>{inline(text)}</h2>")
        elif s.startswith("- "):
            if para:
                flush()
            items.append(s[2:])
        elif not s:
            flush()
        else:
            para.append(s)
    flush()
    return "\n".join(out)


body, figs = [], []
intro_done = False
for cell in nb.cells:
    if cell.cell_type == "markdown":
        if not intro_done:
            intro_done = True
            body.append(f'<div class="intro">{markdown(cell.source)}</div>')
        else:
            body.append(markdown(cell.source))
        continue
    src = cell.source.strip()
    if src.startswith("import plotly.io"):
        continue
    body.append(f'<details class="code"><summary>Show code</summary><pre>{html.escape(src)}</pre></details>')
    for o in cell.outputs:
        if o.output_type == "stream":
            body.append(f'<pre class="out">{html.escape(o.text.strip())}</pre>')
        elif o.output_type in ("display_data", "execute_result"):
            d = o.get("data", {})
            if "application/vnd.plotly.v1+json" in d:
                k = len(figs)
                figs.append(d["application/vnd.plotly.v1+json"])
                body.append(f'<div class="fig" id="fig{k}" role="img" aria-label="interactive 3D view"></div>')
            elif "text/plain" in d:
                body.append(f'<pre class="out">{html.escape(d["text/plain"].strip())}</pre>')

page = f"""<title>OctreeCity Quickstart</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;750&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; 3D views run wide inside it, code folds away. */
:root {{
  --bg: #0f1318; --panel: #161b22; --line: #262e38; --fg: #e6e9ee; --muted: #9aa4b2;
  --accent: #f2a65a; --codebg: #12171d;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  color-scheme: dark;
}}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.6 var(--display); margin: 0; }}
main {{ max-width: 980px; margin: 0 auto; padding-block: 40px 80px; padding-inline: 20px; }}
header h1 {{ font-size: clamp(30px, 5vw, 44px); font-weight: 750; letter-spacing: -0.02em; margin: 0 0 6px; text-wrap: balance; }}
header .meta {{ font: 13px var(--mono); color: var(--muted); }}
.intro p {{ font-size: 18px; color: var(--fg); max-width: 68ch; }}
.intro strong {{ color: var(--accent); font-weight: 600; }}
h2 {{ font-size: 23px; font-weight: 650; margin: 48px 0 6px; display: flex; gap: 12px; align-items: baseline; text-wrap: balance; }}
.step {{ font: 500 13px var(--mono); color: #1b1307; background: var(--accent); border-radius: 4px; padding: 2px 7px; }}
p, li {{ color: var(--muted); max-width: 70ch; }}
p strong, li strong {{ color: var(--fg); }}
code {{ font: 0.9em var(--mono); color: var(--fg); background: var(--codebg); padding: 1px 5px; border-radius: 4px; }}
details.code {{ margin: 14px 0 8px; }}
details.code summary {{ cursor: pointer; font: 500 12.5px var(--mono); color: var(--muted); letter-spacing: 0.04em; }}
details.code summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
pre {{ font: 13px/1.55 var(--mono); background: var(--codebg); border: 1px solid var(--line); border-radius: 6px;
       padding: 12px 14px; overflow-x: auto; margin: 8px 0; }}
pre.out {{ color: #c9d1db; border-left: 3px solid var(--accent); }}
.fig {{ width: 100%; height: 600px; margin: 12px 0 4px; border: 1px solid var(--line); border-radius: 8px;
        overflow: hidden; background: #11151b; }}
.hint {{ font: 12.5px var(--mono); color: var(--muted); }}
footer {{ margin-top: 60px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 13px; color: var(--muted); }}
@media (max-width: 640px) {{ .fig {{ height: 440px; }} main {{ padding-inline: 16px; }} }}
</style>
<main>
<header>
  <h1>OctreeCity quickstart</h1>
  <div class="meta">KAIST campus, Daejeon &middot; notebook run on 13 Oct 2026 &middot; drag to rotate, scroll to zoom</div>
</header>
{''.join(body)}
<footer>Generated from <code>notebooks/octreecity_quickstart.ipynb</code>. Map data © OpenStreetMap contributors (ODbL); building heights: V-World.</footer>
</main>
<script src="{PLOTLY}"></script>
<script>
const FIGS = {json.dumps(figs, separators=(",", ":"))};
FIGS.forEach((f, k) => {{
  const el = document.getElementById("fig" + k);
  const layout = Object.assign({{}}, f.layout, {{autosize: true, height: undefined, width: undefined}});
  Plotly.newPlot(el, f.data, layout, {{responsive: true, displaylogo: false}});
}});
</script>
"""
OUT.write_text(page)
print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB, {len(figs)} 3D views")
