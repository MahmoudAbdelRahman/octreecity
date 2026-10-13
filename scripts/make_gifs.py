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


def to_gif(pattern, out, fps=10, width=640):
    """Frames -> optimised GIF with a shared palette (ffmpeg)."""
    pal = TMP / "palette.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern,
                    "-vf", f"scale={width}:-1:flags=lanczos,palettegen=max_colors=128", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern, "-i", str(pal),
                    "-lavfi", f"scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", str(out)], check=True)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


def frames_dir(name):
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    return d


def sun_day():
    """Shadows over downtown Osaka on 21 December, sunrise to sunset."""
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
    surf = (surface_height(city) - city.origin[2])[lo:hi, lo:hi]
    n = nt * tc.n // every
    tz = timezone(timedelta(hours=9))
    d = frames_dir("sun")
    times = [datetime(2026, 12, 21, 7, 20, tzinfo=tz) + timedelta(minutes=10 * k) for k in range(58)]
    hours = np.zeros(len(pts))
    k = 0
    for t in times:
        az, alt = sun_position(34.69, 135.50, t)
        if alt <= 1:
            continue
        lit = ~shadow(city, pts, az, alt)
        hours += lit * 10 / 60
        g = np.full((n, n), np.nan)
        g[ii, jj] = lit.astype(float)
        fig, ax = plt.subplots(figsize=(6.4, 6.9), facecolor=BG)
        ax.set_facecolor(BG)
        bld = surf[::every, ::every] > 3
        img = np.zeros((n, n, 3))
        img[:] = (0.13, 0.15, 0.19)                       # shade
        img[g == 1] = (0.98, 0.80, 0.45)                  # sunlit street
        hb = np.clip(surf[::every, ::every] / 150, 0, 1)
        img[bld] = (np.array([0.62, 0.65, 0.70])[None] * (0.75 + 0.35 * hb[bld][:, None]))
        ax.imshow(img.transpose(1, 0, 2), origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        compass = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
        ax.set_title(f"Downtown Osaka, 21 December  {t:%H:%M}  sun from the {compass[int((az + 22.5) // 45) % 8]}, {alt:.0f}° high",
                     color=FG, fontsize=11, loc="left")
        ax.text(0, -0.04, "yellow: direct sun on the street   grey: buildings   OctreeCity",
                transform=ax.transAxes, color=MUTED, fontsize=8.5)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05)
        fig.savefig(d / f"f{k:03d}.png", dpi=100, facecolor=BG)
        plt.close(fig)
        k += 1
    to_gif(str(d / "f%03d.png"), OUT / "sun_day.gif", fps=8, width=560)


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
    to_gif(str(d / "f%03d.png"), OUT / "orbit.gif", fps=12, width=520)


def adapt():
    """Clip of the split-and-merge video: geometry, then adaptation rounds."""
    src = ROOT / "outputs/split_merge.mp4"
    d = frames_dir("adapt")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "4", "-t", "21", "-i", str(src),
                    "-vf", "fps=8", str(d / "f%03d.png")], check=True)
    to_gif(str(d / "f%03d.png"), OUT / "adapt.gif", fps=8, width=640)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sun", "orbit", "adapt"]
    for w in which:
        {"sun": sun_day, "orbit": orbit, "adapt": adapt}[w]()
