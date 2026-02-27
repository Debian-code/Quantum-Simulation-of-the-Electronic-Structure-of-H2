import csv
from collections import defaultdict

import matplotlib.pyplot as plt

FIXED_CSV = "fixed_point_results.csv"
APPENDIX_CSV = "appendix_distance_curve.csv"

ANSAZE_ORDER = ["uccsd", "puccsd", "puccd"]
MAPPER_ORDER = ["jordan_wigner", "parity", "bravyi_kitaev"]
OPTIMIZER_ORDER = ["slsqp", "cobyla"]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "distance": float(row["distance_A"]),
                    "ansatz": row["ansatz"],
                    "mapper": row["mapper"],
                    "optimizer": row["optimizer"],
                    "exact": float(row["exact_total_hartree"]),
                    "vqe": float(row["vqe_total_hartree"]),
                    "abs_error": float(row["abs_error_hartree"]),
                    "iterations": int(row["iterations"]),
                    "elapsed_s": float(row["elapsed_s"]),
                }
            )
    return rows


def pretty_mapper(name):
    return {
        "jordan_wigner": "Jordan-Wigner",
        "parity": "Parity",
        "bravyi_kitaev": "Bravyi-Kitaev",
    }[name]


def pretty_optimizer(name):
    return name.upper()


def combo_key(row):
    return (row["mapper"], row["optimizer"])


def combo_label(mapper, optimizer):
    return f"{pretty_mapper(mapper)} + {pretty_optimizer(optimizer)}"


def save_grouped_metric_plot(rows, metric_key, ylabel, title, output_name, log_scale=False):
    rows = [r for r in rows if r["ansatz"] in ANSAZE_ORDER]

    combo_list = [(m, o) for m in MAPPER_ORDER for o in OPTIMIZER_ORDER]
    value_by_triplet = {
        (r["mapper"], r["optimizer"], r["ansatz"]): r[metric_key] for r in rows
    }

    x = list(range(len(combo_list)))
    width = 0.24
    offsets = [-width, 0.0, width]

    plt.figure(figsize=(10.6, 4.9))

    for i, ansatz in enumerate(ANSAZE_ORDER):
        y = [value_by_triplet[(m, o, ansatz)] for (m, o) in combo_list]
        plt.bar([xi + offsets[i] for xi in x], y, width=width, label=ansatz.upper())

    if log_scale:
        plt.yscale("log")

    labels = [combo_label(m, o) for (m, o) in combo_list]
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    plt.legend(ncols=3)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()
    return output_name


def mean(values):
    return sum(values) / len(values) if values else 0.0


def save_mapping_optimizer_heatmaps(rows):
    rows = [r for r in rows if r["ansatz"] in ANSAZE_ORDER]
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["mapper"], r["optimizer"])].append(r)

    metric_specs = [
        ("abs_error", "Mean absolute error (Hartree)", True),
        ("iterations", "Mean iterations", False),
        ("elapsed_s", "Mean runtime (s)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)

    for ax, (metric, title, is_log) in zip(axes, metric_specs):
        matrix = []
        for mapper in MAPPER_ORDER:
            row_vals = []
            for optimizer in OPTIMIZER_ORDER:
                vals = [r[metric] for r in grouped[(mapper, optimizer)]]
                v = mean(vals)
                if is_log:
                    v = max(v, 1e-16)
                row_vals.append(v)
            matrix.append(row_vals)

        if is_log:
            import matplotlib.colors as mcolors

            norm = mcolors.LogNorm(vmin=min(min(r) for r in matrix), vmax=max(max(r) for r in matrix))
            img = ax.imshow(matrix, cmap="viridis", norm=norm)
        else:
            img = ax.imshow(matrix, cmap="viridis")

        ax.set_title(title)
        ax.set_xticks(range(len(OPTIMIZER_ORDER)), labels=[pretty_optimizer(o) for o in OPTIMIZER_ORDER])
        ax.set_yticks(range(len(MAPPER_ORDER)), labels=[pretty_mapper(m) for m in MAPPER_ORDER])

        for i in range(len(MAPPER_ORDER)):
            for j in range(len(OPTIMIZER_ORDER)):
                val = matrix[i][j]
                if metric == "iterations":
                    txt = f"{val:.1f}"
                elif metric == "elapsed_s":
                    txt = f"{val:.2f}"
                else:
                    txt = f"{val:.1e}"
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    out = "global_mapping_optimizer_heatmaps.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_appendix_distance_curve(rows, ansatz):
    rows = sorted(rows, key=lambda r: r["distance"])
    x = [r["distance"] for r in rows]
    exact = [r["exact"] for r in rows]
    vqe = [r["vqe"] for r in rows]

    plt.figure(figsize=(7.6, 4.6))
    plt.plot(x, exact, "k--", marker="o", label="Exact")
    plt.plot(x, vqe, marker="o", label=f"VQE (Jordan-Wigner + SLSQP, {ansatz.upper()})")
    plt.xlabel("H-H distance (Angstrom)")
    plt.ylabel("Total energy (Hartree)")
    plt.title("Supplementary Distance Validation Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = f"appendix_distance_validation_curve_{ansatz}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main():
    fixed_rows = load_rows(FIXED_CSV)
    appendix_rows = load_rows(APPENDIX_CSV)

    outputs = [
        save_grouped_metric_plot(
            fixed_rows,
            metric_key="abs_error",
            ylabel="Absolute error (Hartree)",
            title="Global Accuracy Comparison at 0.735 Angstrom",
            output_name="global_abs_error_by_combo_ansatz.png",
            log_scale=True,
        ),
        save_grouped_metric_plot(
            fixed_rows,
            metric_key="iterations",
            ylabel="Optimizer iterations",
            title="Global Iteration Comparison at 0.735 Angstrom",
            output_name="global_iterations_by_combo_ansatz.png",
            log_scale=False,
        ),
        save_grouped_metric_plot(
            fixed_rows,
            metric_key="elapsed_s",
            ylabel="Runtime (s)",
            title="Global Runtime Comparison at 0.735 Angstrom",
            output_name="global_runtime_by_combo_ansatz.png",
            log_scale=False,
        ),
        save_mapping_optimizer_heatmaps(fixed_rows),
        save_appendix_distance_curve(appendix_rows, appendix_rows[0]["ansatz"]),
    ]

    print("Generated plots:")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
