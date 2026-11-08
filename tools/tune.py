"""Evaluate parameter variants of agent.py against a panel of opponents.

  python tools/tune.py '{"bluff_freq":0.4}' '{"raise_eq":[0.6,0.7,0.74,0.78]}' --deals 400 --seeds 1,2
'{}' = defaults. Prints one line per variant: panel mean and per-opponent bb/deal.
"""
import argparse, json, os, statistics, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfplay

PANEL = ["baseline-bots/baseline_random.py", "baseline-bots/baseline_always_pot.py",
         "baseline-bots/baseline_calling_station.py", "opponents/eq_bot.py",
         "opponents/tight_passive.py", "opponents/lag.py", "opponents/station_plus.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--agent", default="agent.py")
    ap.add_argument("--panel", default=None)
    o = ap.parse_args()
    panel = o.panel.split(",") if o.panel else PANEL
    jobs = []
    for v in o.variants:
        spec = o.agent + ("@" + v if v != "{}" else "")
        for opp in panel:
            for s in o.seeds.split(","):
                jobs.append((spec, opp, s, o.deals))
    with Pool(8) as pool:
        res = pool.map(selfplay.play, jobs)
    k = 0
    for v in o.variants:
        per = []
        for opp in panel:
            vals = []
            for s in o.seeds.split(","):
                vals += res[k][1]; k += 1
            per.append(statistics.mean(vals))
        name = " ".join(f"{os.path.basename(p)[:-3].replace('baseline_','')}={x:+.2f}" for p, x in zip(panel, per))
        print(f"MEAN {statistics.mean(per):+.3f} | {v} | {name}", flush=True)


if __name__ == "__main__":
    main()
