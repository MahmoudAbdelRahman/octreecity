"""The paper's parent-links experiment (train_parents.py, variant B) written with the library.

Cells, parent cells and links come from octreecity (build, parents, links, learn), the model from
octreecity.attention. Same data, split, normalisation, model size, training and scoring as the
paper script, so the scores can be compared with experiments/results/parents/B_*.json.

    --check   copy the weights of the paper's model into LinkModel and compare outputs on test tiles
              (needs the paper scripts folder), then stop.

Needs the paper data (paper-code geo_oct.pkl, urban_wind height pictures) and torch; reads only.
Usage: python scripts/train_parents_lib.py [--seed 0] [--steps 3000] [--out par_lib.json] [--check]
"""
import argparse, json, math, pickle, resource, sys, time

import numpy as np
import torch

from octreecity.attention import LinkModel, masked_mse
from octreecity.build import build
from octreecity.columns import ColumnCity
from octreecity.fill import from_raster
from octreecity.learn import Tokens, pixel_map, reach_m, tokens
from octreecity.links import across, hierarchy, neighbours
from octreecity.rules import mixed
from octreecity.sources import ColumnSource
from octreecity.urban import BUILDING

PC = "/Users/mahmoud/Projects/octreecity-paper-code"
PICS = "/Users/mahmoud/Projects/urban_wind/outputs/height_pictures.npz"
NAMES = ["svf", "sun_hours", "irradiation", "wind_K"]


def h2_of(pic):
    """1 m height picture -> 2 m building flag and mean height ([x, y], y north)."""
    h = pic.astype(np.float64)[::-1, :].T
    b = (h.reshape(128, 2, 128, 2) > 0).mean((1, 3)) >= 0.5
    return np.where(b, h.reshape(128, 2, 128, 2).mean((1, 3)), 0.0), b


def tile_tokens(d, pic):
    h2, b = h2_of(pic)
    city = ColumnCity.from_heightfield(np.zeros_like(h2), h2 / 2, BUILDING, int(math.ceil((h2.max() + 16) / 2)),
                                       voxel_size=2.0)
    tree = build(ColumnSource(city), mixed(), 3).balance(6)
    layer = tree.in_box((0, 0, 0), (128, 128, 1))
    tree.cells["building"] = (tree.label == BUILDING).astype(float)
    tree.cells["h30"] = from_raster(tree, np.where(b, h2, 0.0), "max") / 30
    tree.cells["s16"] = tree.sizes_m() / 16
    tree.cells["y"] = np.full((tree.n_leaves, 4), np.nan)
    tree.cells["y"][layer] = d["vals"]                      # the paper's targets, same leaf order
    p = tree.parents(levels_above=4)                        # up to the whole 256 m tile
    # parent summaries as in the paper: built share and mean building height of the square, size
    x, y, _ = p.corners()
    s = p.sizes()
    pf = {"building": np.zeros(len(p)), "h30": np.zeros(len(p)), "s16": s * 2.0 / 16}
    for k in range(len(p)):
        bb, hh = b[x[k]:x[k] + s[k], y[k]:y[k] + s[k]], h2[x[k]:x[k] + s[k], y[k]:y[k] + s[k]]
        pf["building"][k] = bb.mean()
        pf["h30"][k] = hh[bb].mean() / 30 if bb.any() else 0.0
    n = tree.n_leaves
    links = (neighbours(tree, 6, mask=layer) | hierarchy(tree, p)
             | across(tree, p, leaves=False).where(lambda a, c: c >= n))
    tok = tokens(tree, ["building", "h30", "s16"], links, targets=["y"], select=layer, parents=p,
                 parent_features=pf, k=16)
    tok.offset = tok.offset[..., [0, 1, 3]]                 # 2D task: dx, dy, key size
    tok.extra["pmap"] = pixel_map(tree, layer).ravel()
    tok.extra["n"] = int(layer.sum())
    tok.extra["building"] = d["building"]
    tok.targets[:tok.extra["n"]][d["building"]] = np.nan    # no loss on building cells
    return tok


def score(model, test, data, mu, sd):
    sq, cnt, sqn = np.zeros(4), np.zeros(4), 0.0
    with torch.no_grad():
        for tok, d in zip(test, data):
            pred = model.run(tok.to_torch()).numpy()[:tok.extra["n"]] * sd + mu
            pp = pred[tok.extra["pmap"]].T
            tr = d["px"].reshape(4, -1)
            m = np.isfinite(tr)
            e = np.where(m, pp - tr, 0.0)
            sq += (e ** 2).sum(1)
            cnt += m.sum(1)
            sqn += ((e / sd[:, None]) ** 2).sum()
    return {**{"rmse_" + k: float(np.sqrt(sq[i] / cnt[i])) for i, k in enumerate(NAMES)},
            "rmse_norm_all": float(np.sqrt(sqn / cnt.sum()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default="par_lib.json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = pickle.load(open(f"{PC}/data/geo/geo_oct.pkl", "rb"))
    pics = np.load(PICS)["pictures"]
    split = np.random.default_rng(0).permutation(len(data))         # same split as the paper
    test_d = [data[i] for i in sorted(split[:64])]
    train_d = [data[i] for i in split[64:]]
    t0 = time.time()
    tok = {d["tile"]: tile_tokens(d, pics[d["tile"]]) for d in data}
    build_s = time.time() - t0
    v = np.concatenate([d["vals"][~d["building"]] for d in train_d])
    mu, sd = np.nanmean(v, 0).astype(np.float32), np.nanstd(v, 0).astype(np.float32)
    for t in tok.values():
        t.targets = (t.targets - mu) / sd
    train = [tok[d["tile"]] for d in train_d]
    test = [tok[d["tile"]] for d in test_d]
    print(f"tokens built for {len(tok)} tiles in {build_s:.0f} s", flush=True)

    torch.manual_seed(a.seed)
    model = LinkModel(3, 4, dim=64, heads=4, layers=6, offset_dim=3, n_kinds=0, pos_dims=2, pos_scale=256.0)

    if a.check:
        sys.path.insert(0, f"{PC}/experiments/scripts")
        import train_parents as tp

        ref = tp.Model()
        ref.load_state_dict({k: v for k, v in model.state_dict().items()})
        worst = 0.0
        for d in test_d[:8]:
            g = tp.graph(d, "B", pics[d["tile"]])
            b = tp.batch([d], {d["tile"]: g}, mu, sd)
            with torch.no_grad():
                r = ref(b["feat"], b["xy"], b["idx"], b["off"], b["valid"]).numpy()
                o = model.run(tok[d["tile"]].to_torch()).numpy()
            # match tokens by position and size
            key = lambda xy, s: (round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(s), 3))
            ids = {key(g["xy"][i], g["size"][i]): i for i in range(g["T"])}
            t = tok[d["tile"]]
            m = np.array([ids[key(t.position[i, :2], t.size[i])] for i in range(len(t))])
            worst = max(worst, float(np.abs(o - r[m]).max()))
        print(f"same weights, 8 test tiles: largest output difference {worst:.2e}")
        return

    rng = np.random.default_rng(a.seed)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(a.steps):
        b = Tokens.concat([train[i] for i in rng.choice(len(train), 2, replace=False)]).to_torch()
        loss = masked_mse(model.run(b), b["targets"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(step, round(loss.item(), 4), round(time.time() - t0), flush=True)
    train_s = time.time() - t0
    model.eval()
    res = {"variant": "B-library", "seed": a.seed, "steps": a.steps, **score(model, test, test_d, mu, sd),
           "tokens_per_tile": float(np.mean([len(t) for t in tok.values()])),
           "leaf_tokens_per_tile": float(np.mean([t.extra["n"] for t in tok.values()])),
           "token_build_s": round(build_s), "train_s": round(train_s),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20),
           "reach_area_m_6_layers": float(np.mean([reach_m(t, 6, starts=np.nonzero(
               t.is_leaf & ~np.r_[t.extra["building"], np.zeros(len(t) - t.extra["n"], bool)])[0][:300])
               for t in test[:10]]))}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
