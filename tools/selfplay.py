"""In-process duplicate-match harness (no HTTP): fast self-play and tuning.

Usage:
  python tools/selfplay.py A.py B.py [--seeds 1,2,3] [--deals 1000] [--procs 8]
  A bot spec is a .py file exposing new_half/choose_action/on_hand_end.
  Append @'{"key": value}' to override agent _PARAMS (merged into defaults).
Prints one summary line per match plus a pooled line.
"""
import argparse, importlib.util, itertools, json, math, os, statistics, sys, time
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from game import Action, AgentConfig, Game, make_deal, button_for_deal, default_action  # noqa

_counter = itertools.count()


def load(spec):
    path, _, params = spec.partition("@")
    p = os.path.join(ROOT, path) if not os.path.isabs(path) else path
    name = f"bot_{next(_counter)}_{os.getpid()}"
    sp = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    if params:
        import strategy
        P = dict(strategy.DEFAULT_PARAMS)
        P.update(json.loads(params))
        mod._PARAMS = P
    return mod


class Handle:
    def __init__(self, mod):
        self.mod, self.mem, self.errors, self.think, self.calls, self.maxcall = mod, {}, 0, 0.0, 0, 0.0
        self.pending = []

    def start(self):
        self.mem.clear()
        self.mod.new_half(AgentConfig(), self.mem)

    def act(self, obs):
        t = time.perf_counter()
        try:
            for r in self.pending:
                self.mod.on_hand_end(r, self.mem)
            self.pending = []
            a = self.mod.choose_action(obs, self.mem)
            a = Action(a)
            if a not in obs.legal_actions:
                raise ValueError("illegal")
        except Exception:
            self.errors += 1
            a = default_action(obs.to_call)
        dt = time.perf_counter() - t
        self.think += dt; self.calls += 1; self.maxcall = max(self.maxcall, dt)
        return a


def play(args):
    a_spec, b_spec, seed, deals = args
    A, B = Handle(load(a_spec)), Handle(load(b_spec))
    nets = [[0] * deals, [0] * deals]
    stats = []
    for half in (1, 2):
        bots = (A, B) if half == 1 else (B, A)
        for h in (A, B):
            h.errors = 0; h.think = 0.0; h.calls = 0; h.maxcall = 0.0; h.pending = []
            h.start()
        for d in range(deals):
            g = Game(make_deal(str(seed), d), button_for_deal(d), hand_index=d)
            while not g.is_terminal:
                s = g.to_act
                g.apply(bots[s].act(g.observation(s)))
            a_seat = 0 if half == 1 else 1
            nets[half - 1][d] = g.result().nets[a_seat]
            for s in (0, 1):
                bots[s].pending.append(g.hand_result(s))
        for h in (A, B):
            for r in h.pending:
                try:
                    h.mod.on_hand_end(r, h.mem)
                except Exception:
                    h.errors += 1
        stats.append((A.think, A.calls, A.errors, A.maxcall, B.think, B.errors))
    paired = [(nets[0][d] + nets[1][d]) / 2.0 for d in range(deals)]
    return seed, paired, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=8)
    o = ap.parse_args()
    seeds = [s for s in o.seeds.split(",")]
    jobs = [(o.a, o.b, s, o.deals) for s in seeds]
    with Pool(min(o.procs, len(jobs))) as pool:
        res = pool.map(play, jobs)
    allp = []
    tag = f"{o.a.split('/')[-1]} vs {o.b.split('/')[-1]}"
    for seed, paired, stats in res:
        m = statistics.mean(paired); se = statistics.stdev(paired) / math.sqrt(len(paired))
        allp += paired
        th = max(s[0] for s in stats); er = sum(s[2] for s in stats); mx = max(s[3] for s in stats)
        ber = sum(s[5] for s in stats)
        v = "WIN" if m - 1.96 * se > 0 else ("LOSS" if m + 1.96 * se < 0 else "CHOP")
        print(f"seed {seed}: {m:+.3f} ± {1.96*se:.3f} bb/deal {v} | A think/half max {th:.2f}s maxcall {mx*1000:.1f}ms err {er} | B err {ber}")
    m = statistics.mean(allp); se = statistics.stdev(allp) / math.sqrt(len(allp))
    print(f"POOLED {tag}: {m:+.3f} ± {1.96*se:.3f} bb/deal over {len(allp)} deals")


if __name__ == "__main__":
    main()
