"""Animated GIFs for the README: shadows through a day, a 3D orbit, adaptive cells.

Usage: .venv/bin/python scripts/make_gifs.py [sun|orbit|adapt ...]
"""

import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/gifs"
OUT.mkdir(parents=True, exist_ok=True)
BG, FG, MUTED = "#0f1318", "#e6e9ee", "#8b95a3"
TMP = ROOT / "outputs/gifs/_frames"


def to_gif(pattern, out, fps=10, width=640, colors=64, every=1):
    """Frames -> small GIF: shared palette built from the changing pixels,
    only the changed rectangle stored per frame, optional frame skipping."""
    pal = TMP / "palette.png"
    pick = f"select='not(mod(n\\,{every}))',setpts=N/FRAME_RATE/TB," if every > 1 else ""
    scale = f"{pick}scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern,
                    "-vf", f"{scale},palettegen=max_colors={colors}:stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern, "-i", str(pal),
                    "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
                    "-r", str(fps), "-loop", "0", str(out)], check=True)
    print(out, f"{out.stat().st_size / 1e6:.2f} MB")


def encode_only():
    """Re-encode the three GIFs from frames already on disk (no re-rendering)."""
    to_gif(str(TMP / "sun/f%03d.png"), OUT / "sun_day.gif", fps=5, width=460, colors=96, every=2)
    to_gif(str(TMP / "orbit/f%03d.png"), OUT / "orbit.gif", fps=8, width=460, colors=48, every=2)
    to_gif(str(TMP / "adapt/f%03d.png"), OUT / "adapt.gif", fps=6, width=520, colors=48)


def frames_dir(name):
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    return d


def sun_day():
    """Downtown Osaka on 21 December: shadows move through the day while the
    total hours of direct sun build up, ending on the full heatmap."""
    from octreecity.analysis import shadow, street_points, sun_position, surface_height
    from octreecity.tiles import TiledCity

    tc = TiledCity(ROOT / "data/osaka_1m.ocz")
    M, nt, i0, j0 = 1, 3, 40, 46
    city = tc.region(i0 - M, j0 - M, i0 + nt + M, j0 + nt + M)
    every = 2
    pts, (ii, jj) = street_points(city, every=every)
    lo, hi = M * tc.n, (M + nt) * tc.n
    keep = (ii >= lo) & (ii < hi) & (jj >= lo) & (jj < hi)
    pts, ii, jj = pts[keep], (ii[keep] - lo) // every, (jj[keep] - lo) // every
    surf = (surface_height(city) - city.origin[2])[lo:hi, lo:hi][::every, ::every]
    n = nt * tc.n // every
    tz = timezone(timedelta(hours=9))
    step = 10
    times = [datetime(2026, 12, 21, 7, 0, tzinfo=tz) + timedelta(minutes=step * k) for k in range(64)]
    states = []
    hours = np.zeros(len(pts))
    for t in times:
        az, alt = sun_position(34.69, 135.50, t)
        if alt <= 1:
            continue
        lit = ~shadow(city, pts, az, alt)
        hours += lit * step / 60
        states.append((t, az, alt, lit.copy(), hours.copy()))
    total = states[-1][4]
    vmax = np.ceil(total.max())
    cmap = plt.get_cmap("inferno")
    bld = surf > 3
    hb = np.clip(surf / 150, 0, 1)
    bcol = np.array([0.62, 0.65, 0.70])[None] * (0.75 + 0.35 * hb[bld][:, None])
    compass = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    d = frames_dir("sun")

    def frame(k, acc, lit, title, note, final=False):
        g = np.full((n, n), np.nan)
        g[ii, jj] = acc
        img = cmap(np.clip(np.nan_to_num(g) / vmax, 0, 1))[..., :3]
        img[np.isnan(g)] = (0.13, 0.15, 0.19)
        if lit is not None:                        # streets in sun right now glow warm white
            on = np.full((n, n), False)
            on[ii, jj] = lit
            img[on] = img[on] * 0.45 + np.array([1.0, 0.93, 0.72]) * 0.55
        img[bld] = bcol
        fig = plt.figure(figsize=(6.4, 7.2), facecolor=BG)
        ax = fig.add_axes([0.02, 0.1, 0.96, 0.82])
        ax.imshow(img.transpose(1, 0, 2), origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(0.03, 0.955, title, color=FG, fontsize=11.5, weight="bold" if final else "normal")
        fig.text(0.03, 0.925, note, color=MUTED, fontsize=9)
        cax = fig.add_axes([0.06, 0.055, 0.88, 0.018])
        cax.imshow(cmap(np.linspace(0, 1, 256))[None], aspect="auto", extent=(0, vmax, 0, 1))
        cax.set_yticks([])
        cax.tick_params(colors=MUTED, labelsize=8, length=0)
        for sp in cax.spines.values():
            sp.set_visible(False)
        fig.text(0.06, 0.018, "hours of direct sun so far" if not final else "total hours of direct sun, 21 December",
                 color=MUTED, fontsize=8.5)
        fig.savefig(d / f"f{k:03d}.png", dpi=100, facecolor=BG)
        plt.close(fig)

    k = 0
    for t, az, alt, lit, acc in states:
        frame(k, acc, lit, f"Downtown Osaka, 21 December  {t:%H:%M}",
              f"sun from the {compass[int((az + 22.5) // 45) % 8]}, {alt:.0f} deg high; glow = in sun now")
        k += 1
    for _ in range(24):                            # hold the result for 3 s
        frame(k, total, None, "Hours of direct sun on the shortest day",
              f"downtown Osaka at street level, every 2 m; from {len(states)} sun positions", final=True)
        k += 1
    to_gif(str(d / "f%03d.png"), OUT / "sun_day.gif", fps=5, width=460, colors=96, every=2)


def orbit():
    """Slow 3D orbit around a block of downtown Osaka (merged OBJ faces)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from octreecity.exports import surface_quads
    from octreecity.octree import BUILDING
    from octreecity.tiles import TiledCity

    c = TiledCity(ROOT / "data/osaka_1m.ocz").region(41, 47, 43, 49)
    q, _ = surface_quads(c, [BUILDING])
    q = q - np.array(c.origin)
    up = np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0])[:, 2] > -1e-9
    q = q[up]
    nrm = np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    h = q[:, :, 2].mean(1)
    base = np.array([0.80, 0.82, 0.86])
    warm = np.array([0.98, 0.72, 0.42])
    tint = np.clip(h / 160, 0, 1)[:, None]
    d = frames_dir("orbit")
    for k, az in enumerate(np.arange(0, 360, 6)):
        light = np.array([np.sin(np.radians(az + 200)), np.cos(np.radians(az + 200)), 0.9])
        light /= np.linalg.norm(light)
        sh = np.clip(0.4 + 0.65 * (nrm @ light), 0.22, 1.1)
        fc = np.clip(((1 - 0.5 * tint) * base + 0.5 * tint * warm) * sh[:, None], 0, 1)
        fig = plt.figure(figsize=(6.4, 5.2), facecolor=BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG)
        ax.add_collection3d(Poly3DCollection(q, facecolors=fc, edgecolors=fc, linewidths=0.1))
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_zlim(0, 220)
        ax.set_box_aspect((512, 512, 220))
        ax.view_init(elev=28, azim=az - 60)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.text(0.03, 0.94, "Downtown Osaka, 512 m, from PLATEAU at 1 m", color=FG, fontsize=10)
        fig.savefig(d / f"f{k:03d}.png", dpi=100, facecolor=BG)
        plt.close(fig)
    to_gif(str(d / "f%03d.png"), OUT / "orbit.gif", fps=8, width=460, colors=48, every=2)


def adapt():
    """Clip of the split-and-merge video: geometry, then adaptation rounds."""
    src = ROOT / "outputs/split_merge.mp4"
    d = frames_dir("adapt")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "4", "-t", "21", "-i", str(src),
                    "-vf", "fps=8", str(d / "f%03d.png")], check=True)
    to_gif(str(d / "f%03d.png"), OUT / "adapt.gif", fps=6, width=520, colors=48)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sun", "orbit", "adapt"]
    for w in which:
        {"sun": sun_day, "orbit": orbit, "adapt": adapt, "encode": encode_only}[w]()
