"""Simulate a slow tournament CPU: slow down equity() by a factor and check the time guard.

  python tools/slow_test.py [slow_factor] [opponent.py]
Reports the judge-style charged time estimate assuming plumbing also scales.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selfplay
from game import Game, make_deal

factor = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
opp = sys.argv[2] if len(sys.argv) > 2 else "baseline-bots/baseline_always_pot.py"
import strategy
orig = strategy.equity

def slow_equity(*a, **k):
    t = time.perf_counter()
    r = orig(*a, **k)
    time.sleep((time.perf_counter() - t) * (factor - 1))
    return r

strategy.equity = slow_equity
strategy.SPEED = factor * 0.8
A = selfplay.Handle(selfplay.load("agent.py")); B = selfplay.Handle(selfplay.load(opp))
A.start(); B.start()
charged = 0.0
dts = []
orig_act = A.act
def act(obs):
    t = time.perf_counter(); a = orig_act(obs); dt = time.perf_counter() - t
    dts.append(dt)
    return a
A.act = act
for d in range(1000):
    g = Game(make_deal("slow", d), d % 2, hand_index=d); bots = (A, B)
    while not g.is_terminal:
        s = g.to_act; g.apply(bots[s].act(g.observation(s)))
    for s in (0, 1): bots[s].pending.append(g.hand_result(s))
plumb = 0.0005 * factor  # dev-laptop round trip+plumbing ~0.4-0.5 ms (judge report), scaled
real = sum(max(0.0, dt + plumb - 0.001) for dt in dts)
print(f"factor {factor}: calls {len(dts)} est.judge-charged {real:.2f}s agent-estimate {A.mem['charged']:.2f}s "
      f"final n {A.mem['n']} max call {max(dts)*1000:.1f}ms errors {A.errors + A.mem['errors']}")
