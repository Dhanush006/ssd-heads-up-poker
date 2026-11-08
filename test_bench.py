import math, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import bench

T0, T1 = bench.TRAIN[0], bench.TRAIN[1]
TP = "opponents/tight_passive.py"
V1 = "opponents/v1/agent_v1.py"
TILT = "opponents/tilt.py"


def test_ci():
    m, h = bench.ci([1, 2, 3, 4])
    assert m == 2.5
    assert abs(h - 1.96 * statistics.stdev([1, 2, 3, 4]) / math.sqrt(4)) < 1e-12
    assert bench.ci([5])[1] == float("inf")


def test_name():
    assert bench.name("opponents/v1/agent_v1.py") == "agent_v1"
    assert bench.name("baseline-bots/baseline_random.py") == "random"
    assert bench.name('agent.py@{"a": 1}') == "agent"
    assert bench.name('opponents/v21/agent_v21.py@{"x": "y.py"}') == "agent_v21"


def test_hard_names_exist():
    names = {bench.name(p) for p in bench.TRAIN + bench.HOLDOUT}
    assert bench.HARD <= names


def test_fixed_load_pins_budgets():
    m = bench._fixed_load("opponents/v21/agent_v21.py")
    assert m._SOFT_BUDGET >= 1e9 and m._HARD_BUDGET >= 1e9
    assert bench.PINNED["opponents/v21/agent_v21.py"] is True


def test_fixed_load_leaves_other_modules():
    m = bench._fixed_load("opponents/eq_bot.py")
    assert not hasattr(m, "_SOFT_BUDGET")
    assert bench.PINNED["opponents/eq_bot.py"] is False


OPPS = ["a.py", "b.py"]
SEEDS = ["1", "2"]


def fake(offsets, add):
    # per-opponent offset so cand-base differs by opponent; deals vary
    return {o: ({s: [float(i) + k + off + add.get(o, 0.0) for i in range(5)]
                 for k, s in enumerate(SEEDS)}, 1.0, 0)
            for o, off in zip(OPPS, offsets)}


def test_summarize_asymmetric():
    base = fake([0.0, 0.0], {})
    # a: +1 on every deal; b: +3 on deals alternating +2/+4 (varies within opponent)
    cand = fake([0.0, 0.0], {"a.py": 1.0})
    for s in SEEDS:
        cand["b.py"][0][s] = [x + (2.0 if i % 2 == 0 else 4.0)
                              for i, x in enumerate(base["b.py"][0][s])]
    rows, panel = bench.summarize(cand, base, OPPS, SEEDS)
    ra, rb = rows
    assert abs(ra[3] - 1.0) < 1e-9 and ra[4] == 0.0
    assert abs(rb[3] - 2.8) < 1e-9
    assert abs(rb[4] - 1.96 * statistics.stdev([2, 4, 2, 4, 2] * 2) / math.sqrt(10)) < 1e-9
    # per-deal panel = mean(1, d_b): over 10 deals
    pv = [1.5, 2.5, 1.5, 2.5, 1.5] * 2
    assert abs(panel[0] - 1.9) < 1e-9
    assert abs(panel[1] - 1.96 * statistics.stdev(pv) / math.sqrt(10)) < 1e-9
    assert len(rows[0]) == 9 and ra[0] == "a.py" and ra[1] - ra[2] == ra[3]


def test_per_seed_panel():
    base = fake([0.0, 0.0], {})
    cand = fake([0.0, 0.0], {"a.py": 1.0})
    cand["b.py"][0]["2"] = [x + 3.0 for x in cand["b.py"][0]["2"]]
    vals, (m, h) = bench.per_seed_panel(cand, base, OPPS, SEEDS)
    # seed 1: mean(1, 0)=0.5 ; seed 2: mean(1, 3)=2.0
    assert vals == [0.5, 2.0]
    assert abs(m - 1.25) < 1e-12
    assert abs(h - 1.96 * statistics.stdev([0.5, 2.0]) / math.sqrt(2)) < 1e-12


def row(o, cm=1.0, bm=0.0, dm=1.0, dc=0.1, ct=1.0, bt=1.0, ce=0, be=0):
    return (o, cm, bm, dm, dc, ct, bt, ce, be)


def good():
    train = [row(TP), row(T0)]
    hold = [row(V1), row(TILT)]
    return train, (1.0, 0.1), hold, (1.0, 0.1)


def res(g):
    return [x[1] for x in g]


def test_gates_all_pass():
    assert all(res(bench.gates(*good())))


def test_g1():
    t, pt, h, ph = good()
    assert res(bench.gates(t, pt, h, (0.1, 0.5)))[0] is False


def test_g2():
    t, pt, h, ph = good()
    t2 = [row(r[0], cm=-1.0, bm=0.0) for r in t]
    assert res(bench.gates(t2, pt, h, ph))[1] is False
    t3 = [row(r[0], cm=2.0, bm=2.0) for r in t]
    assert res(bench.gates(t3, pt, h, ph))[1] is True


def test_g3():
    t, pt, h, ph = good()
    h2 = [row(TILT, dm=-0.6, dc=0.0)] + h[:1]
    assert res(bench.gates(t, pt, h2, ph))[2] is False
    h3 = [row(TILT, dm=-0.5, dc=0.0)] + h[:1]
    assert res(bench.gates(t, pt, h3, ph))[2] is True


def test_g4():
    t, pt, h, ph = good()
    t2 = [row(TP, dm=-0.3, dc=0.1)] + t[1:]
    g = res(bench.gates(t2, pt, h, ph))
    assert g[3] is False and g[2] is True
    t3 = [row(TP, dm=-0.1, dc=0.1)] + t[1:]
    assert res(bench.gates(t3, pt, h, ph))[3] is True


def test_g5():
    t, pt, h, ph = good()
    t2 = [row(t[0][0], ce=1)] + t[1:]
    assert res(bench.gates(t2, pt, h, ph))[4] is False


def test_g6_total():
    t, pt, h, ph = good()
    # total base 4.0; cand 1.1x passes, 1.2x fails
    def with_think(c):
        return ([row(r[0], ct=c) for r in t], pt, [row(r[0], ct=c) for r in h], ph)
    assert res(bench.gates(*with_think(1.1)))[5] is True
    assert res(bench.gates(*with_think(1.2)))[5] is False
    # per-row spike allowed when total is fine
    t2 = [row(t[0][0], ct=1.5, bt=1.0), row(t[1][0], ct=0.7, bt=1.0)]
    assert res(bench.gates(t2, pt, h, ph))[5] is True


def test_run_smoke_identical_bots_zero_diff():
    spec = "opponents/eq_bot.py"
    cand, cpin = bench.run(spec, [spec], ["1"], 2, 1)
    base, bpin = bench.run(spec, [spec], ["1"], 2, 1)
    rows, panel = bench.summarize(cand, base, [spec], ["1"])
    assert rows[0][3] == 0.0 and panel[0] == 0.0
    assert cpin is False


def test_panels_disjoint_and_sized():
    assert len(bench.TRAIN) == 14 and len(bench.HOLDOUT) == 6
    assert not set(bench.TRAIN) & set(bench.HOLDOUT)
    assert {bench.name(p) for p in bench.HOLDOUT} == {
        "sizer", "river_bluffer", "trapper", "adaptive_caller", "nut_nit", "range_eq"}
