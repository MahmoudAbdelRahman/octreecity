"""Illustration video: error-based merging of cells on a real wind field.

2D quadtree for readability (the octree does the same with 8 children).
Field: pedestrian wind ratio K at 1.75 m, urban_wind tile 110, wind from
270 deg (light CFD preset), averaged to 4 m cells.

Rule: four sibling cells merge into their parent when the parent's value
(the mean) differs from every fine value it replaces by at most tol.
Building cells merge only with building cells (geometry is a hard limit).
Cell sizes stay between min_cell and max_cell.

Usage: .venv/bin/python scripts/video_error_merge.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow, Rectangle
from PIL import Image

SRC = Path.home() / "Projects/urban_wind/cfd/cases/tile_0110_from270_light/frame_ratio.png"
OUT = Path(__file__).resolve().parent.parent / "outputs/error_merge.mp4"

N, CELL, LEVELS, TOL = 64, 4, 3, 0.20
BG, PANEL, FG, MUTED = "#0f1318", "#161b22", "#e6e9ee", "#8b95a3"
GOOD, BAD, ACCENT, BLD = "#7bd389", "#f2705a", "#f2a65a", "#262b33"
CMAP = plt.get_cmap("viridis")
KMAX = 1.8
FPS = 30

# ------------------------------------------------------------------ field

a = np.array(Image.open(SRC)).astype(np.float64)
K = (a[..., 0] * 256 + a[..., 1]) / 10000
inside = a[..., 2] == 255
kb = K.reshape(N, CELL, N, CELL)
ib = inside.reshape(N, CELL, N, CELL)
BUILDING = ib.mean(axis=(1, 3)) >= 0.5
FINE = np.nanmean(np.where(ib, np.nan, kb), axis=(1, 3))
FINE[BUILDING] = np.nan


def block_stats(level):
    s = 1 << level
    B = FINE.reshape(N // s, s, N // s, s)
    bld = BUILDING.reshape(N // s, s, N // s, s)
    all_b = bld.all(axis=(1, 3))
    any_b = bld.any(axis=(1, 3))
    with np.errstate(all="ignore"):
        mean = np.nanmean(B, axis=(1, 3))
        err = np.nanmax(np.abs(B - mean[:, None, :, None]), axis=(1, 3))
    return mean, err, all_b, any_b


STATS = {l: block_stats(l) for l in range(1, LEVELS + 1)}


def merged_masks(tol, upto):
    """merged[l][i, j]: block (i, j) at level l is one leaf."""
    merged = {}
    for l in range(1, upto + 1):
        mean, err, all_b, any_b = STATS[l]
        ok = all_b | (~any_b & (err <= tol))
        if l > 1:
            c = merged[l - 1]
            ok &= c[0::2, 0::2] & c[1::2, 0::2] & c[0::2, 1::2] & c[1::2, 1::2]
        merged[l] = ok
    return merged


def leaves(merged, upto):
    """List of (i, j, size) in fine-cell units, plus the value image."""
    taken = np.zeros((N, N), bool)
    img = FINE.copy()
    out = []
    for l in range(upto, 0, -1):
        s = 1 << l
        mean = STATS[l][0]
        for i, j in zip(*np.nonzero(merged[l])):
            if taken[i * s, j * s]:
                continue
            taken[i * s:(i + 1) * s, j * s:(j + 1) * s] = True
            img[i * s:(i + 1) * s, j * s:(j + 1) * s] = mean[i, j]
            out.append((i * s, j * s, s))
    for i, j in zip(*np.nonzero(~taken)):
        out.append((i, j, 1))
    return out, img


def rect_segments(lv):
    segs = []
    for i, j, s in lv:
        x0, y0, x1, y1 = j, i, j + s, i + s
        segs += [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)],
                 [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
    return segs


def rgba(img):
    rgb = CMAP(np.clip(img / KMAX, 0, 1))
    rgb[np.isnan(img)] = matplotlib.colors.to_rgba(BLD)
    return rgb


# ------------------------------------------------------------------ drawing

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
ax = fig.add_axes([0.035, 0.07, 0.48, 0.853])
side = fig.add_axes([0.56, 0.0, 0.44, 1.0])
cax = fig.add_axes([0.035, 0.03, 0.48, 0.016])
plt.rcParams.update({"font.family": "DejaVu Sans"})


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def draw_map(img, segs, alpha_lines=1.0, extra_segs=None, extra_alpha=0.0, overlays=()):
    ax.cla()
    ax.set_facecolor(BG)
    ax.imshow(rgba(img), origin="upper", extent=(0, N, N, 0), interpolation="nearest")
    ax.add_collection(LineCollection(segs, colors=(1, 1, 1, 0.55 * alpha_lines), linewidths=0.6))
    if extra_segs:
        ax.add_collection(LineCollection(extra_segs, colors=(1, 1, 1, 0.55 * extra_alpha), linewidths=0.6))
    for patch in overlays:
        ax.add_patch(patch)
    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, pad=2)
    ax.text(-0.5, -1.4, "wind", color=MUTED, fontsize=9, ha="left")
    ax.add_patch(FancyArrow(5.2, -1.75, 5, 0, width=0.35, head_width=1.3, head_length=1.2,
                            color=MUTED, clip_on=False, length_includes_head=True))


def draw_colorbar():
    cax.cla()
    cax.imshow(CMAP(np.linspace(0, 1, 256))[None], aspect="auto", extent=(0, KMAX, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1.0, 1.5], ["0", "0.5", "1.0", "1.5"], color=MUTED, fontsize=8)
    cax.tick_params(length=0, pad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.text(0, 1.6, "wind ratio K at 1.75 m  ·  dark = building  ·  tile 256 m wide",
             color=MUTED, fontsize=8, ha="left")


def draw_side(title, body, stats=None, step=None, inset=None):
    side.cla()
    side.set_facecolor(PANEL)
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis("off")
    side.add_patch(Rectangle((0, 0), 1, 1, color=PANEL, zorder=-1))
    if step:
        side.text(0.07, 0.9, step, color=ACCENT, fontsize=10, family="DejaVu Sans Mono")
    side.text(0.07, 0.84, title, color=FG, fontsize=19, weight="bold", va="top")
    side.text(0.07, 0.75, body, color=MUTED, fontsize=11.5, va="top", linespacing=1.55)
    if stats:
        y = 0.2
        for k, v in stats:
            side.text(0.07, y, k, color=MUTED, fontsize=10)
            side.text(0.93, y, v, color=FG, fontsize=13, ha="right", family="DejaVu Sans Mono")
            y -= 0.055
    if inset:
        inset()


def group_inset(vals, t, verdict, tol):
    """2x2 numbers at the side; t in [0, 1] animates the merge."""
    x0, y0, w = 0.2, 0.33, 0.13
    mean = np.mean(vals)
    err = np.max(np.abs(vals - mean))
    good = err <= tol
    m = ease(t) if good else 0
    for k, v in enumerate(vals):
        r, c = divmod(k, 2)
        val = (1 - m) * v + m * mean
        side.add_patch(Rectangle((x0 + c * w, y0 + (1 - r) * w), w, w,
                                 facecolor=CMAP(np.clip(val / KMAX, 0, 1)),
                                 edgecolor=(1, 1, 1, 0.8 * (1 - m)), lw=1.2))
        side.text(x0 + c * w + w / 2, y0 + (1 - r) * w + w / 2, f"{v:.2f}",
                  color="white" if val < 0.9 else "#111", fontsize=12, ha="center",
                  va="center", family="DejaVu Sans Mono", alpha=1 - m)
    side.add_patch(Rectangle((x0, y0), 2 * w, 2 * w, fill=False,
                             edgecolor=GOOD if good else BAD, lw=2.5))
    if m > 0:
        side.text(x0 + w, y0 + w, f"{mean:.2f}", color="white" if mean < 0.9 else "#111",
                  fontsize=15, ha="center", va="center", family="DejaVu Sans Mono", alpha=m)
    if verdict:
        side.text(0.62, 0.52, f"mean   {mean:.2f}", color=FG, fontsize=11, family="DejaVu Sans Mono")
        side.text(0.62, 0.47, f"error  {err:.2f}", color=FG, fontsize=11, family="DejaVu Sans Mono")
        side.text(0.62, 0.42, f"tol    {tol:.2f}", color=FG, fontsize=11, family="DejaVu Sans Mono")
        side.text(0.62, 0.35, "merge" if good else "keep", color=GOOD if good else BAD,
                  fontsize=16, weight="bold")


# ------------------------------------------------------------------ scenes

scenes = []


def scene(seconds):
    def deco(f):
        scenes.append((int(seconds * FPS), f))
        return f
    return deco


fine_leaves, _ = leaves({}, 0)
fine_segs = rect_segments(fine_leaves)


@scene(4)
def title(t):
    draw_map(FINE, fine_segs, alpha_lines=ease(t * 3))
    draw_side("Error-based merging",
              "Merge neighbouring cells only where the\nfield they carry barely changes.\n\n"
              "Here: pedestrian wind ratio K at 1.75 m\nfrom a CFD run (Daejeon tile 110,\nwind from the west), in 2D for clarity.",
              step="OctreeCity · adaptive meshing")


@scene(4)
def start(t):
    draw_map(FINE, fine_segs)
    draw_side("Start at the finest size",
              f"Every cell is {CELL} m (min cell size).\nThe largest allowed cell is "
              f"{CELL << LEVELS} m (max cell size).\nBoth are user settings.",
              stats=[("cells", f"{N * N:,}"), ("tolerance", f"{TOL:.2f}")], step="step 0")


def pick_group(good):
    mean, err, all_b, any_b = STATS[1]
    cand = ~any_b & ((err <= TOL * 0.5) if good else (err > TOL * 2.5))
    ii, jj = np.nonzero(cand)
    k = np.argmin((ii - 24) ** 2 + (jj - (12 if good else 30)) ** 2)
    i, j = ii[k], jj[k]
    return i, j, FINE[2 * i:2 * i + 2, 2 * j:2 * j + 2].ravel()


for good in (True, False):
    gi, gj, gv = pick_group(good)

    def make(gi=gi, gj=gj, gv=gv, good=good):
        @scene(6)
        def example(t):
            hi = Rectangle((2 * gj - 0.2, 2 * gi - 0.2), 2.4, 2.4, fill=False,
                           edgecolor=GOOD if good else BAD, lw=2.5)
            draw_map(FINE, fine_segs, overlays=[hi])
            draw_side("One group of four" if good else "A group at a shear layer",
                      "Replace four cells by their mean.\nError = largest difference between\n"
                      "the mean and the values it replaces." if good else
                      "Near buildings the wind changes fast.\nThe mean would hide that, so the\n"
                      "four cells stay separate.",
                      step="the rule",
                      inset=lambda: group_inset(gv, (t - 0.45) / 0.35, t > 0.3, TOL))
    make()

prev_leaves, prev_img = fine_leaves, FINE
for lvl in range(1, LEVELS + 1):
    merged = merged_masks(TOL, lvl)
    new_leaves, new_img = leaves(merged, lvl)
    s = 1 << lvl
    mean, err, all_b, any_b = STATS[lvl]
    if lvl == 1:
        cand = np.ones_like(all_b)
    else:
        c = merged_masks(TOL, lvl - 1)[lvl - 1]
        cand = c[0::2, 0::2] & c[1::2, 0::2] & c[0::2, 1::2] & c[1::2, 1::2]
    show = cand & ~all_b
    ok = merged[lvl]

    def make(prev_leaves=prev_leaves, prev_img=prev_img, new_leaves=new_leaves,
             new_img=new_img, s=s, show=show, ok=ok, lvl=lvl):
        prev_segs, new_segs = rect_segments(prev_leaves), rect_segments(new_leaves)
        n_ok, n_bad = int((show & ok).sum()), int((show & ~ok).sum())

        @scene(5.5)
        def level(t):
            hl = min(1, t / 0.35) * (1 - ease((t - 0.75) / 0.25))
            over = []
            for i, j in zip(*np.nonzero(show)):
                col = GOOD if ok[i, j] else BAD
                over.append(Rectangle((j * s, i * s), s, s, facecolor=col, alpha=0.38 * hl,
                                      edgecolor=col, lw=1.2))
            m = ease((t - 0.42) / 0.33)
            img = np.where(np.isnan(prev_img), np.nan, (1 - m) * np.nan_to_num(prev_img)
                           + m * np.nan_to_num(new_img))
            draw_map(img, new_segs, extra_segs=prev_segs, extra_alpha=1 - m, overlays=over)
            n = len(prev_leaves) + m * (len(new_leaves) - len(prev_leaves))
            draw_side(f"Level {lvl}: try {CELL * s} m cells",
                      "Every group of four leaves is tested.\n"
                      "Green: error within tolerance, merge.\n"
                      "Red: error too large, or the group\nmixes building and air: keep.",
                      stats=[("groups merged", f"{n_ok:,}"), ("groups kept", f"{n_bad:,}"),
                             ("cells", f"{int(round(n)):,}")],
                      step=f"step {lvl} of {LEVELS}")
    make()
    prev_leaves, prev_img = new_leaves, new_img

final_leaves, final_img = prev_leaves, prev_img


@scene(4)
def result(t):
    draw_map(final_img, rect_segments(final_leaves))
    sizes = np.bincount([s for _, _, s in final_leaves], minlength=(1 << LEVELS) + 1)
    draw_side("Result",
              f"Small cells where the wind changes,\nbig cells where it is calm.\n\n"
              f"Every merged cell is within {TOL:.2f} of the\nfine values it replaced.",
              stats=[("cells", f"{len(final_leaves):,}  ({N * N / len(final_leaves):.1f}x fewer)"),
                     ("4 / 8 / 16 / 32 m", " / ".join(str(sizes[1 << l]) for l in range(LEVELS + 1)))],
              step="done")


TOLS = np.concatenate([np.geomspace(0.02, 0.6, 150), np.geomspace(0.6, TOL, 90)])


@scene(len(TOLS) / FPS)
def sweep(t):
    tol = TOLS[min(int(t * len(TOLS)), len(TOLS) - 1)]
    lv, img = leaves(merged_masks(tol, LEVELS), LEVELS)
    draw_map(img, rect_segments(lv))
    side_bar = lambda: (side.add_patch(Rectangle((0.07, 0.36), 0.86, 0.025, color="#1e252e")),
                        side.add_patch(Rectangle((0.07, 0.36), 0.86 * np.log(tol / 0.02) / np.log(0.6 / 0.02),
                                                 0.025, color=ACCENT)),
                        side.text(0.07, 0.40, "tolerance", color=MUTED, fontsize=10),
                        side.text(0.93, 0.40, f"{tol:.2f}", color=FG, fontsize=12, ha="right",
                                  family="DejaVu Sans Mono"))
    draw_side("One knob: the tolerance",
              "Tight tolerance: fine everywhere, accurate.\n"
              "Loose tolerance: fewer cells, smoother field.\n"
              "Cell sizes never leave the min/max range.\n\n"
              "At street level the wind changes almost\n"
              "everywhere, so savings here are modest.",
              stats=[("cells", f"{len(lv):,}"), ("fewer than fine", f"{N * N / len(lv):.1f}x")],
              step="trade-off", inset=side_bar)


@scene(5)
def outro(t):
    draw_map(final_img, rect_segments(final_leaves))
    draw_side("Same idea in 3D",
              "Octree: 8 children instead of 4.\n\n"
              "Geometry is a hard limit: building and\nair never merge.\n"
              "Min/max cell size bound every cell.\n"
              "Neighbours keep the 2:1 size rule.\n"
              "Above the roofs the flow is smooth, so\n"
              "most of the merging should happen there.\n\n"
              "The field can come from a coarse CFD run\n"
              "or the ML surrogate, then the mesh is\nadapted and solved again.",
              step="next")


# ------------------------------------------------------------------ render

if __name__ == "__main__":
    import sys
    OUT.parent.mkdir(exist_ok=True)
    draw_colorbar()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
        for k, (n, f) in enumerate(scenes):
            f(0.6)
            fig.savefig(Path(sys.argv[1]) / f"scene{k:02d}.png", dpi=60, facecolor=BG)
        sys.exit()
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=2500,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    total = sum(n for n, _ in scenes)
    print(f"{total} frames, {total / FPS:.1f} s")
    with writer.saving(fig, str(OUT), dpi=100):
        for n, f in scenes:
            for k in range(n):
                f(k / max(1, n - 1))
                writer.grab_frame(facecolor=BG)
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
