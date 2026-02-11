import os, sys, csv, time, math
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.data_generator import generate_dronz_2d
from src.task1_2d.closest_pair_dc import closest_pair_2d

def baseline_bruteforce_2d(dronz):
    best = None
    best_d = float("inf")
    n = len(dronz)
    for i in range(n):
        for j in range(i + 1, n):
            dx = dronz[i][1] - dronz[j][1]
            dy = dronz[i][2] - dronz[j][2]
            d = math.hypot(dx, dy)
            ids = tuple(sorted((dronz[i][0], dronz[j][0])))
            if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and (best is None or ids < tuple(sorted((best[0][0], best[1][0]))))):
                best = (dronz[i], dronz[j])
                best_d = d
    return best, best_d

def time_avg(fn, dronz, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(dronz)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return mean(times)

def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)

    # Optimized tested on larger sizes; baseline only small sizes
    N_OPT = [1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000, 10_000_000]
    N_BASE = [1_000, 2_000, 3_000]

    SEED = 42
    REPEATS = 3

    rows = []
    for n in N_OPT:
        dronz = generate_dronz_2d(n, bound=1000, seed=SEED)

        opt_t = time_avg(closest_pair_2d, dronz, repeats=REPEATS)
        base_t = ""
        if n in N_BASE:
            base_t = time_avg(baseline_bruteforce_2d, dronz, repeats=REPEATS)

        rows.append((n, base_t, opt_t))
        print(f"n={n:>8}  baseline={base_t if base_t!='' else '—':>10}  optimized={opt_t:.6f}s")

    out_csv = os.path.join(ROOT, "results", "task1_times.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "baseline_seconds", "optimized_seconds"])
        w.writerows(rows)

    import matplotlib.pyplot as plt
    ns = [r[0] for r in rows]
    opt = [r[2] for r in rows]

    plt.figure()
    plt.plot(ns, opt, marker="o", label="Optimized (Divide & Conquer)")

    ns_b = [r[0] for r in rows if r[1] != ""]
    b = [r[1] for r in rows if r[1] != ""]
    if ns_b:
        plt.plot(ns_b, b, marker="o", label="Baseline (O(n^2))")

    plt.xlabel("Number of dronz (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Task 1 (2D): Runtime Comparison")
    plt.grid(True)
    plt.legend()

    out_png = os.path.join(ROOT, "plots", "task1_times.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
