"""Generate the opponent-model dataset.

  python tools/gen_opp_data.py --opponents 150 --deals 400 --seeds 11,12 --out data/opp_dataset.npz

Rows: one per (opponent, checkpoint). X = features at that checkpoint (profile-0 run),
Y = mean paired bb/deal of each profile against that opponent, G = opponent id.
Tuning seeds 1-4 and benchmark seeds 101-104 are NOT used here.
The agent's time guard is pinned (bench._fixed_load) so labels are reproducible.
"""
import argparse, json, os, statistics, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "opponents"))
import selfplay                      # noqa: E402
import bench                         # noqa: E402
import param_bot                     # noqa: E402
import oppmodel                      # noqa: E402
from game import Game, make_deal, button_for_deal   # noqa: E402

CKPTS = (40, 80, 160, 320)
FIXED = ["baseline-bots/baseline_random.py", "baseline-bots/baseline_always_pot.py",
         "baseline-bots/baseline_calling_station.py", "opponents/eq_bot.py",
         "opponents/tight_passive.py", "opponents/lag.py", "opponents/station_plus.py"]


def build_descs(n_opponents, rng_seed=2026):
    rng = np.random.default_rng(rng_seed)
    descs = [("fixed", p, 0) for p in FIXED]
    for i in range(n_opponents):
        descs.append(("param", param_bot.sample(rng), i))
    return descs


def make_opponent(desc):
    kind, payload, seed = desc
    if kind == "param":
        return selfplay.Handle(param_bot.make(payload, seed))
    return selfplay.Handle(selfplay.load(payload))


def run_match(job):
    desc, profile_idx, overrides, seed, deals = job
    spec = "agent.py" + ("@" + json.dumps(overrides) if overrides else "")
    A = selfplay.Handle(bench._fixed_load(spec))   # pinned time guard -> reproducible
    B = make_opponent(desc)
    nets, feats = [], []
    for half in (1, 2):
        bots = (A, B) if half == 1 else (B, A)
        for h in (A, B):
            h.pending = []
            h.start()
        half_nets = []
        for d in range(deals):
            g = Game(make_deal(str(seed), d), button_for_deal(d), hand_index=d)
            while not g.is_terminal:
                s = g.to_act
                g.apply(bots[s].act(g.observation(s)))
            half_nets.append(g.result().nets[0 if half == 1 else 1])
            for s in (0, 1):
                bots[s].pending.append(g.hand_result(s))
            if half == 1 and profile_idx == 0 and (d + 1) in CKPTS:
                # Results are delivered at the next act, so the model has seen d
                # (not d+1) hands here. Key the snapshot by d+1 but pass the model's
                # own count to features(), matching how the runtime calls it.
                m = A.mem["model"]
                feats.append((d + 1, oppmodel.features(m, len(m["results"]))))
        nets.append(half_nets)
    paired = [(nets[0][d] + nets[1][d]) / 2.0 for d in range(deals)]
    return desc, profile_idx, seed, statistics.mean(paired), feats


def aggregate(res, K, ckpts=CKPTS):
    """res: list of (desc, k, seed, mean, feats). Returns X (rows,8), Y (rows,K), G (rows,)."""
    order, means, feats = [], {}, {}
    for desc, k, seed, mean, fl in res:
        key = json.dumps(desc, sort_keys=True)
        if key not in order:
            order.append(key)
        means.setdefault((key, k), []).append(mean)
        for c, x in fl:
            feats.setdefault((key, c), []).append(np.asarray(x, dtype=float))
    X, Y, Gr = [], [], []
    for gi, key in enumerate(order):
        y = [float(np.mean(means[(key, k)])) for k in range(K)]
        for c in ckpts:
            if (key, c) in feats:
                X.append(np.mean(feats[(key, c)], axis=0))
                Y.append(y)
                Gr.append(gi)
    return np.array(X), np.array(Y), np.array(Gr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents", type=int, default=150)
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--seeds", default="11,12")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "opp_dataset.npz"))
    ap.add_argument("--procs", type=int, default=8)
    o = ap.parse_args()
    seeds = o.seeds.split(",")
    with open(os.path.join(ROOT, "data", "profiles.json")) as f:
        overrides = json.load(f)["overrides"]
    K = len(overrides)
    descs = build_descs(o.opponents)
    jobs = [(d, k, overrides[k], s, o.deals) for d in descs for k in range(K) for s in seeds]
    with Pool(max(1, min(o.procs, len(jobs)))) as pool:
        res = pool.map(run_match, jobs, chunksize=4)
    X, Y, G = aggregate(res, K)
    np.savez(o.out, X=X, Y=Y, G=G)
    print(f"saved {len(X)} rows from {len(descs)} opponents, K={K}")


if __name__ == "__main__":
    main()
