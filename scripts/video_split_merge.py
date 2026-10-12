"""Illustration video: the CFD-style split-and-merge loop (adapt()) in 2D.

Same field as video_error_merge.py: pedestrian wind ratio K at 1.75 m,
urban_wind tile 110, wind from 270 deg, on 4 m cells. The "solve" is
replaced by sampling that CFD result (mean over each cell).

Loop, mirroring octreecity.adapt.adapt():
  start at max cell size, split cells that mix building and air (geometry);
  then repeat: solve, jump to each face neighbour, split air cells whose
  largest jump > tol (+ 1 buffer layer), merge sibling groups of 4 whose
  jumps are all < tol / 2, enforce the 2:1 rule; stop when nothing changes.

Usage: .venv/bin/python -W ignore scripts/video_split_merge.py [stills_dir]
"""

import sys
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
OUT = Path(__file__).resolve().parent.parent / "outputs/split_merge.mp4"

N, CELL, SMAX, TOL = 64, 4, 8, 0.30
BG, PANEL, FG, MUTED = "#0f1318", "#161b22", "#e6e9ee", "#8b95a3"
SPLIT, MERGE, ACCENT, BLD, EMPTY = "#f2705a", "#5aa9f2", "#f2a65a", "#262b33", "#1b222b"
CMAP = plt.get_cmap("viridis")
KMAX, FPS = 1.8, 30

# ------------------------------------------------------------------ field

a = np.array(Image.open(SRC)).astype(np.float64)
K = (a[..., 0] * 256 + a[..., 1]) / 10000
inside = a[..., 2] == 255
kb, ib = K.reshape(N, CELL, N, CELL), inside.reshape(N, CELL, N, CELL)
BUILDING = ib.mean(axis=(1, 3)) >= 0.5
with np.errstate(all="ignore"):
    FINE = np.nanmean(np.where(ib, np.nan, kb), axis=(1, 3))
FINE[BUILDING] = np.nan

# ------------------------------------------------------------------ 2D quadtree (leaves: (i, j, s) in 4 m units)


def label(i, j, s):
    b = BUILDING[i:i + s, j:j + s]
    return 1 if b.all() else 0 if not b.any() else -1


def owner_map(leaves):
    g = np.empty((N, N), np.int64)
    for k, (i, j, s) in enumerate(leaves):
        g[i:i + s, j:j + s] = k
    return g


def solve(leaves):
    """Stand-in for a CFD solve: mean of the fine field over each cell."""
    v = np.full(len(leaves), np.nan)
    for k, (i, j, s) in enumerate(leaves):
        blk = FINE[i:i + s, j:j + s]
        if not np.isnan(blk).all():
            v[k] = np.nanmean(blk)
    return v


def neighbour_pairs(g):
    pairs = set()
    for a_, b_ in ((g[1:], g[:-1]), (g[:, 1:], g[:, :-1])):
        d = a_ != b_
        pairs |= set(zip(a_[d].tolist(), b_[d].tolist()))
    return np.array(sorted(pairs))


def indicator(leaves, v, g):
    ind = np.zeros(len(leaves))
    p = neighbour_pairs(g)
    ok = ~np.isnan(v[p[:, 0]]) & ~np.isnan(v[p[:, 1]])
    p = p[ok]
    j = np.abs(v[p[:, 0]] - v[p[:, 1]])
    np.maximum.at(ind, p[:, 0], j)
    np.maximum.at(ind, p[:, 1], j)
    return ind


def children(i, j, s):
    h = s // 2
    return [(i, j, h), (i + h, j, h), (i, j + h, h), (i + h, j + h, h)]


def balance(leaves):
    while True:
        g = owner_map(leaves)
        size = np.array([s for _, _, s in leaves])
        p = neighbour_pairs(g)
        bad = set()
        for x, y in p:
            if size[x] > 2 * size[y]:
                bad.add(x)
            elif size[y] > 2 * size[x]:
                bad.add(y)
        if not bad:
            return leaves
        leaves = [c for k, l in enumerate(leaves) for c in (children(*l) if k in bad else [l])]


def adapt_step(leaves, v):
    g = owner_map(leaves)
    ind = indicator(leaves, v, g)
    air = np.array([label(*l) == 0 for l in leaves])
    size = np.array([s for _, _, s in leaves])
    split = air & (ind > TOL) & (size > 1)
    p = neighbour_pairs(g)
    buf = np.zeros_like(split)
    buf[p[:, 1][split[p[:, 0]]]] = True
    buf[p[:, 0][split[p[:, 1]]]] = True
    split |= buf & air & (size > 1)
    calm = air & (ind < TOL / 2) & ~split
    index = {l: k for k, l in enumerate(leaves)}
    groups = []
    for k, (i, j, s) in enumerate(leaves):
        S = 2 * s
        if S > SMAX or i % S or j % S:
            continue
        sib = [index.get(c) for c in children(i, j, S)]
        if None not in sib and all(calm[q] for q in sib):
            groups.append(sib)
    gone = {q for grp in groups for q in grp}
    new = []
    for k, l in enumerate(leaves):
        if split[k]:
            new += children(*l)
        elif k not in gone:
            new.append(l)
    for grp in groups:
        i, j, s = leaves[grp[0]]
        new.append((i, j, 2 * s))
    new = balance(new)
    return new, split, groups


def max_error(leaves, v):
    img = paint(leaves, v)
    return np.nanmax(np.abs(img - FINE))


def paint(leaves, v):
    img = np.full((N, N), np.nan)
    for (i, j, s), val in zip(leaves, v):
        img[i:i + s, j:j + s] = val
    img[BUILDING] = np.nan
    return img


# geometry phase: top-down from 32 m cells
geo_steps = [[(i, j, SMAX) for i in range(0, N, SMAX) for j in range(0, N, SMAX)]]
while True:
    cur = geo_steps[-1]
    if all(label(*l) != -1 for l in cur):
        break
    geo_steps.append([c for l in cur for c in (children(*l) if label(*l) == -1 else [l])])

rounds = []
leaves = balance(geo_steps[-1])
v = solve(leaves)
for _ in range(10):
    new, split, groups = adapt_step(leaves, v)
    rounds.append(dict(before=leaves, v=v, split=split, groups=groups, after=new, v_after=solve(new)))
    if sorted(new) == sorted(leaves):
        break
    leaves, v = new, solve(new)
FINAL, FINAL_V = leaves, v

# ------------------------------------------------------------------ drawing

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
ax = fig.add_axes([0.035, 0.07, 0.48, 0.853])
side = fig.add_axes([0.56, 0.0, 0.44, 1.0])
cax = fig.add_axes([0.035, 0.03, 0.48, 0.016])


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def segs(lv):
    out = []
    for i, j, s in lv:
        x0, y0, x1, y1 = j, i, j + s, i + s
        out += [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)], [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
    return out


def rgba(img, field_alpha=1.0):
    rgb = CMAP(np.clip(np.nan_to_num(img) / KMAX, 0, 1))
    empty = np.array(matplotlib.colors.to_rgba(EMPTY))
    rgb = field_alpha * rgb + (1 - field_alpha) * empty
    rgb[np.isnan(img)] = empty
    rgb[BUILDING] = matplotlib.colors.to_rgba(BLD)
    return rgb


def draw_map(img, lines, field_alpha=1.0, old_lines=None, old_alpha=0.0, overlays=()):
    ax.cla()
    ax.set_facecolor(BG)
    ax.imshow(rgba(img, field_alpha), origin="upper", extent=(0, N, N, 0), interpolation="nearest")
    ax.add_collection(LineCollection(segs(lines), colors=(1, 1, 1, 0.55), linewidths=0.6))
    if old_lines:
        ax.add_collection(LineCollection(segs(old_lines), colors=(1, 1, 1, 0.55 * old_alpha), linewidths=0.6))
    for p in overlays:
        ax.add_patch(p)
    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.5, -1.4, "wind", color=MUTED, fontsize=9)
    ax.add_patch(FancyArrow(5.2, -1.75, 5, 0, width=0.35, head_width=1.3, head_length=1.2,
                            color=MUTED, clip_on=False, length_includes_head=True))


def draw_colorbar():
    cax.imshow(CMAP(np.linspace(0, 1, 256))[None], aspect="auto", extent=(0, KMAX, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1.0, 1.5], ["0", "0.5", "1.0", "1.5"], color=MUTED, fontsize=8)
    cax.tick_params(length=0, pad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.text(0, 1.6, "wind ratio K at 1.75 m  ·  dark = building  ·  tile 256 m wide",
             color=MUTED, fontsize=8)


def draw_side(title, body, stats=None, step=None, extra=None):
    side.cla()
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis("off")
    side.add_patch(Rectangle((0, 0), 1, 1, color=PANEL, zorder=-1))
    if step:
        side.text(0.07, 0.9, step, color=ACCENT, fontsize=10, family="DejaVu Sans Mono")
    side.text(0.07, 0.84, title, color=FG, fontsize=19, weight="bold", va="top")
    side.text(0.07, 0.75, body, color=MUTED, fontsize=11.5, va="top", linespacing=1.55)
    if stats:
        y = 0.25
        for k, v in stats:
            side.text(0.07, y, k, color=MUTED, fontsize=10)
            side.text(0.93, y, v, color=FG, fontsize=13, ha="right", family="DejaVu Sans Mono")
            y -= 0.055
    if extra:
        extra()


def legend():
    for k, (c, t) in enumerate(((SPLIT, "split: jump to a neighbour > tol"),
                                (MERGE, "merge: all four calm (< tol / 2)"))):
        y = 0.43 - k * 0.05
        side.add_patch(Rectangle((0.07, y), 0.035, 0.025, color=c, alpha=0.7))
        side.text(0.125, y + 0.004, t, color=FG, fontsize=10.5)


LOOP = ["solve", "estimate error (jumps)", "split / merge", "2:1 rule, min/max size"]


def loop_diagram(active):
    cx, cy, r = 0.5, 0.42, 0.15
    for k, name in enumerate(LOOP):
        ang = np.pi / 2 - k * np.pi / 2
        x, y = cx + r * 1.55 * np.cos(ang), cy + r * np.sin(ang)
        on = k == active
        side.text(x, y, name, color=ACCENT if on else MUTED, fontsize=12 if on else 10.5,
                  weight="bold" if on else "normal", ha="center", va="center")
    for k in range(4):
        a0 = np.pi / 2 - k * np.pi / 2 - 0.45
        a1 = a0 - np.pi / 2 + 0.9
        th = np.linspace(a0, a1, 20)
        side.plot(cx + r * 1.1 * np.cos(th), cy + r * 0.75 * np.sin(th), color="#3a4452", lw=1.5)


scenes = []


def scene(seconds):
    def deco(f):
        scenes.append((max(1, int(seconds * FPS)), f))
        return f
    return deco


@scene(4)
def title(t):
    draw_map(np.full((N, N), np.nan), geo_steps[0])
    draw_side("Split and merge",
              "How CFD codes adapt a mesh, applied to\nthe same wind field as before.\n\n"
              "Start coarse. Split where the solution\njumps between neighbours. Merge where\nit is calm. Repeat.",
              step="OctreeCity · adaptive meshing")


for k in range(1, len(geo_steps)):
    def make(prev=geo_steps[k - 1], cur=geo_steps[k], k=k):
        mixed = [l for l in prev if label(*l) == -1]

        @scene(2.2)
        def geo(t):
            hl = 1 - ease((t - 0.6) / 0.4)
            over = [Rectangle((j, i), s, s, facecolor=SPLIT, alpha=0.35 * hl, edgecolor=SPLIT, lw=1)
                    for i, j, s in mixed]
            m = ease((t - 0.35) / 0.4)
            draw_map(np.full((N, N), np.nan), cur if m > 0.5 else prev, overlays=over)
            draw_side("Step 0: geometry",
                      f"Start at the max cell size ({SMAX * CELL} m).\n"
                      "Split every cell that mixes building and\nair, down to the min size "
                      f"({CELL} m).\nNo field needed yet.",
                      stats=[("cells", f"{len(cur if m > 0.5 else prev):,}")], step="before any solve")
    make()


@scene(3.5)
def first_solve(t):
    r0 = rounds[0]
    draw_map(paint(r0["before"], r0["v"]), r0["before"], field_alpha=ease(t / 0.6))
    draw_side("First solve",
              "Solve the flow on this coarse mesh.\n\n"
              "Here the solve is replaced by the CFD\nresult averaged over each cell. In\n"
              "practice: a coarse OpenFOAM run or the\nML surrogate.",
              stats=[("cells", f"{len(r0['before']):,}"),
                     ("largest error", f"{max_error(r0['before'], r0['v']):.2f}")], step="solve")


for r, R in enumerate(rounds):
    if r == len(rounds) - 1 and not R["split"].any() and not R["groups"]:
        break

    def make(R=R, r=r):
        before, after = R["before"], R["after"]
        img0, img1 = paint(before, R["v"]), paint(after, R["v_after"])
        n_split, n_merge = int(R["split"].sum()), len(R["groups"])
        split_boxes = [before[k] for k in np.nonzero(R["split"])[0]]
        merge_boxes = [(before[g[0]][0], before[g[0]][1], 2 * before[g[0]][2]) for g in R["groups"]]
        err0, err1 = max_error(before, R["v"]), max_error(after, R["v_after"])

        @scene(5.5)
        def rnd(t):
            hl = min(1, t / 0.3) * (1 - ease((t - 0.7) / 0.3))
            over = [Rectangle((j, i), s, s, facecolor=SPLIT, alpha=0.45 * hl, edgecolor=SPLIT, lw=0.8)
                    for i, j, s in split_boxes]
            over += [Rectangle((j, i), s, s, facecolor=MERGE, alpha=0.5 * hl, edgecolor=MERGE, lw=1.2)
                     for i, j, s in merge_boxes]
            m = ease((t - 0.4) / 0.3)
            img = np.where(np.isnan(img0) | np.isnan(img1), np.nan,
                           (1 - m) * np.nan_to_num(img0) + m * np.nan_to_num(img1))
            draw_map(img, after if m > 0.5 else before, old_lines=before if m > 0.5 else None,
                     old_alpha=1 - m, overlays=over)
            n = len(before) + m * (len(after) - len(before))
            draw_side(f"Round {r + 1}",
                      "Compare each cell with its neighbours.\n"
                      f"Tolerance: {TOL:.2f} in K. Then solve again\non the new mesh.",
                      stats=[("cells split", f"{n_split:,}"), ("groups merged", f"{n_merge:,}"),
                             ("cells", f"{int(round(n)):,}"),
                             ("largest error", f"{(1 - m) * err0 + m * err1:.2f}")],
                      step=f"round {r + 1}", extra=legend)
    make()

n_rounds = sum(1 for R in rounds if R["split"].any() or R["groups"])
sizes = np.bincount([s for _, _, s in FINAL], minlength=SMAX + 1)


@scene(4.5)
def converged(t):
    draw_map(paint(FINAL, FINAL_V), FINAL)
    draw_side("Converged",
              f"After {n_rounds} rounds no cell needs splitting\nor merging. Every jump between "
              f"neighbours\nis below {TOL:.2f}, unless the cell is already\nat the min size.",
              stats=[("cells", f"{len(FINAL):,}  ({N * N / len(FINAL):.1f}x fewer than 4 m)"),
                     ("4 / 8 / 16 / 32 m", " / ".join(str(sizes[s]) for s in (1, 2, 4, 8))),
                     ("largest error", f"{max_error(FINAL, FINAL_V):.2f}")],
              step="done")


@scene(8)
def loop(t):
    draw_map(paint(FINAL, FINAL_V), FINAL)
    draw_side("The loop", "What a CFD code with adaptive meshing\nrepeats until the mesh stops changing:",
              step="summary", extra=lambda: loop_diagram(int(t * 8) % 4))


@scene(6)
def why(t):
    draw_map(paint(FINAL, FINAL_V), FINAL)
    draw_side("Why split, not merge?",
              "Merging from the finest size needs the\nfine solution first: the expensive part.\n\n"
              "Splitting needs only a coarse solve, and\nthe work grows with the cells you keep.\n\n"
              "The cost: here it keeps more cells than\nmerging did at a similar error, because\n"
              "of the buffer layer, the 2:1 rule and a\nsimple jump test.\n\n"
              "In OctreeCity: adapt(tree, values, tol).",
              step="next")


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    draw_colorbar()
    print("rounds", n_rounds, "final cells", len(FINAL), "sizes", sizes[[1, 2, 4, 8]].tolist())
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
        d.mkdir(parents=True, exist_ok=True)
        for k, (n, f) in enumerate(scenes):
            f(0.6)
            fig.savefig(d / f"s{k:02d}.png", dpi=60, facecolor=BG)
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
