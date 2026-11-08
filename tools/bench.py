"""Paired benchmark: candidate vs baseline against the same opponents on the same deals.

  python tools/bench.py agent_candidate.py opponents/v21/agent_v21.py --panel all --seeds 101,102,103,104 --deals 1000

Time guard is pinned so runs are reproducible; real time budgets are tested in
test_agent.py / test_oppmodel.py.

G6 measures unguarded compute under the same parallel load (baseline and candidate run
in the same session), not judge-charged time. The per-deal CI assumes deal independence and
is somewhat optimistic, hence the per-seed cross-check printed under the gates.

Panels: TRAIN is 14 bots (the original 7 plus the former hold-out 7, all fair game for
tuning). HOLDOUT is 6 fresh bots (sizer, river_bluffer, trapper, adaptive_caller, nut_nit,
range_eq) used only for final acceptance, never for tuning or training.

Prints per-opponent bb/deal for both bots, the paired difference (cand - base) with a 95% CI,
panel means, and the acceptance gates from the plan. Seeds must never be used for tuning.
"""
import argparse, math, os, statistics, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfplay

TRAIN = ["baseline-bots/baseline_random.py", "baseline-bots/baseline_always_pot.py",
         "baseline-bots/baseline_calling_station.py", "opponents/eq_bot.py",
         "opponents/tight_passive.py", "opponents/lag.py", "opponents/station_plus.py",
         "baseline-bots/baseline_always_half.py", "opponents/v1/agent_v1.py",
         "opponents/check_raiser.py", "opponents/three_bettor.py", "opponents/mixed_nit.py",
         "opponents/tilt.py", "opponents/exploiter.py"]
HOLDOUT = ["opponents/sizer.py", "opponents/river_bluffer.py", "opponents/trapper.py",
           "opponents/adaptive_caller.py", "opponents/nut_nit.py", "opponents/range_eq.py"]
HARD = {"tight_passive", "eq_bot", "agent_v1"}
MAX_REGRESSION = 0.5          # bb/deal, per opponent
THINK_RATIO = 1.15            # candidate total think vs baseline total


def name(path):
    base = os.path.splitext(os.path.basename(path.partition("@")[0]))[0]
    return base.replace("baseline_", "")


assert HARD <= {name(p) for p in TRAIN + HOLDOUT}, "HARD names must exist in the panels"


def ci(vals):
    m = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else float("inf")
    return m, 1.96 * se


_orig_load = selfplay.load
PINNED = {}




def _fixed_load(spec):
    mod = _orig_load(spec)
    pinned = hasattr(mod, "_SOFT_BUDGET") and hasattr(mod, "_HARD_BUDGET")
    if pinned:
        mod._SOFT_BUDGET = 1e9
        mod._HARD_BUDGET = 1e9
    PINNED[spec] = pinned
    return mod


def play_fixed(job):
    selfplay.load = _fixed_load
    res = selfplay.play(job)
    return res, PINNED.get(job[0], False)


def run(spec, opps, seeds, deals, procs):
    jobs = [(spec, o, s, deals) for o in opps for s in seeds]
    with Pool(max(1, min(procs, len(jobs)))) as pool:
        raw = pool.map(play_fixed, jobs)
    res = [r for r, _ in raw]
    pinned = all(p for _, p in raw)
    out, k = {}, 0
    for o in opps:
        paired, think, errs = {}, 0.0, 0
        for s in seeds:
            seed, p, stats = res[k]; k += 1
            paired[s] = p
            think = max(think, max(st[0] for st in stats))
            errs += sum(st[2] for st in stats)
        out[o] = (paired, think, errs)
    return out, pinned


def summarize(cand, base, opps, seeds):
    rows, per_opp_diffs = [], []
    for o in opps:
        cp, ct, ce = cand[o]; bp, bt, be = base[o]
        diffs, cvals, bvals = [], [], []
        for s in seeds:
            diffs += [a - b for a, b in zip(cp[s], bp[s])]
            cvals += cp[s]; bvals += bp[s]
        dm, dc = ci(diffs)
        rows.append((o, statistics.mean(cvals), statistics.mean(bvals), dm, dc, ct, bt, ce, be))
        per_opp_diffs.append(diffs)
    panel = [statistics.mean(x) for x in zip(*per_opp_diffs)]      # per-deal mean over opponents
    return rows, ci(panel)


def per_seed_panel(cand, base, opps, seeds):
    """Panel paired diff (mean over opponents and deals) for each seed, plus mean and CI."""
    vals = []
    for s in seeds:
        vals.append(statistics.mean(
            statistics.mean(a - b for a, b in zip(cand[o][0][s], base[o][0][s])) for o in opps))
    return vals, ci(vals)


def gates(rows_train, panel_train, rows_hold, panel_hold):
    g = []
    g.append(("G1 hold-out paired CI lower bound > 0", panel_hold[0] - panel_hold[1] > 0,
              f"{panel_hold[0]:+.3f} ± {panel_hold[1]:.3f}"))
    ct = statistics.mean(r[1] for r in rows_train); bt = statistics.mean(r[2] for r in rows_train)
    g.append(("G2 train-panel mean >= baseline", ct >= bt, f"cand {ct:+.3f} vs base {bt:+.3f}"))
    worst = min((r[3], name(r[0])) for r in rows_train + rows_hold)
    g.append((f"G3 no opponent regresses > {MAX_REGRESSION}", worst[0] >= -MAX_REGRESSION,
              f"worst {worst[1]} {worst[0]:+.3f}"))
    hard = [r for r in rows_train + rows_hold if name(r[0]) in HARD]
    bad = [name(r[0]) for r in hard if r[3] + r[4] < 0]
    g.append(("G4 hard opponents improve or tie", not bad, f"significantly worse: {bad or 'none'}"))
    errs = sum(r[7] for r in rows_train + rows_hold)
    g.append(("G5 zero candidate errors", errs == 0, f"{errs} errors"))
    rows = rows_train + rows_hold
    ct = sum(r[5] for r in rows); bt = sum(r[6] for r in rows)
    g.append(("G6 total think within 1.15x baseline (same run; real budgets tested in pytest)",
              ct <= THINK_RATIO * bt, f"cand {ct:.1f}s vs base {bt:.1f}s total"))
    return g


def show(title, rows, panel):
    print(f"\n== {title} ==")
    print(f"{'opponent':26s} {'cand':>8s} {'base':>8s} {'diff ± CI':>16s} {'think c/b (s)':>14s} {'err c/b':>8s}")
    for o, cm, bm, dm, dc, ct, bt, ce, be in rows:
        print(f"{name(o):26s} {cm:+8.2f} {bm:+8.2f} {dm:+8.2f} ± {dc:4.2f} {ct:6.2f}/{bt:5.2f} {ce:4d}/{be:<3d}")
    print(f"{'PANEL paired diff':26s} {'':8s} {'':8s} {panel[0]:+8.2f} ± {panel[1]:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"); ap.add_argument("base")
    ap.add_argument("--panel", choices=["train", "holdout", "all"], default="all")
    ap.add_argument("--seeds", default="101,102,103,104")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=8)
    o = ap.parse_args()
    seeds = o.seeds.split(",")
    sets = {"train": TRAIN, "holdout": HOLDOUT}
    todo = ["train", "holdout"] if o.panel == "all" else [o.panel]
    results, seedvals = {}, {}
    for key in todo:
        opps = sets[key]
        cand, cpin = run(o.cand, opps, seeds, o.deals, o.procs)
        base, bpin = run(o.base, opps, seeds, o.deals, o.procs)
        if not (cpin and bpin):
            print(f"WARNING: time guard NOT pinned (candidate pinned={cpin}, baseline pinned={bpin}); "
                  "runs may not be reproducible or paired")
        rows, panel = summarize(cand, base, opps, seeds)
        show(key.upper(), rows, panel)
        results[key] = (rows, panel)
        seedvals[key] = per_seed_panel(cand, base, opps, seeds)
    if o.panel == "all":
        print("\n== ACCEPTANCE GATES (user approval still required) ==")
        g = gates(*results["train"], *results["holdout"])
        for label, ok, detail in g:
            print(f"{'PASS' if ok else 'FAIL'}  {label}  [{detail}]")
        for key in ("train", "holdout"):
            v, (m, h) = seedvals[key]
            print(f"{key} per-seed panel diffs: {[round(x, 3) for x in v]} ({m:+.3f} ± {h:.3f})")
        print("ALL GATES PASS" if all(x[1] for x in g) else "NOT ACCEPTED")


if __name__ == "__main__":
    main()
