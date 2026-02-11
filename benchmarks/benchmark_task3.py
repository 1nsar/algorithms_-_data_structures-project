import os, sys, csv, time, math, heapq
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.data_generator import generate_dronz_3d
from src.task3_topk.topk_kdtree import top_k_pairs

def baseline_topk_allpairs(dronz, k):
    pairs = []
    n = len(dronz)
    for i in range(n):
        for j in range(i + 1, n):
            dx = dronz[i][1] - dronz[j][1]
            dy = dronz[i][2] - dronz[j][2]
            dz = dronz[i][3] - dronz[j][3]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            pairs.append((d, (dronz[i], dronz[j])))
    return heapq.nsmallest(k, pairs, key=lambda x: (x[0], x[1][0][0], x[1][1][0]))

def time_avg(fn, *args, repeats=3, **kwargs):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return mean(times)

def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)

    # Optimized can go larger; baseline keep small and honest
    N_OPT = [1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000, 10_000_000]
    N_BASE = [500, 1_000]

    K_VALUES = [1, 5, 10, 50, 100]
    SEED = 42
    REPEATS = 3

    rows = []
    for n in N_OPT:
        dronz = generate_dronz_3d(n, bound=1000, seed=SEED)
        for k in K_VALUES:
            opt_t = time_avg(top_k_pairs, dronz, k, repeats=REPEATS)

            base_t = ""
            if n in N_BASE and k <= 50:
                base_t = time_avg(baseline_topk_allpairs, dronz, k, repeats=REPEATS)

            rows.append((n, k, base_t, opt_t))
            print(f"n={n:>7} k={k:>4}  baseline={base_t if base_t!='' else '—':>10}  optimized={opt_t:.6f}s")

    out_csv = os.path.join(ROOT, "results", "task3_times.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "k", "baseline_seconds", "optimized_seconds"])
        w.writerows(rows)

    import matplotlib.pyplot as plt
    plt.figure()

    for k in K_VALUES:
        ns = [r[0] for r in rows if r[1] == k]
        opt = [r[3] for r in rows if r[1] == k]
        plt.plot(ns, opt, marker="o", label=f"Optimized k={k}")

    plt.xlabel("Number of dronz (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Task 3 (3D): Optimized Top-k Runtime vs n")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)

    out_png = os.path.join(ROOT, "plots", "task3_times.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_png}")
    print("Note: baseline timings are only collected for small n to avoid excessive runtimes.")

if __name__ == "__main__":
    main()