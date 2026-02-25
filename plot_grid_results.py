import csv

import matplotlib.pyplot as plt

FIXED_CSV = "fixed_point_results.csv"
APPENDIX_CSV = "appendix_distance_curve.csv"


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


def combo_label(row):
    return f"{row['mapper']} + {row['optimizer']}"


def save_fixed_error_plot(rows):
    labels = [combo_label(r) for r in rows]
    values = [r["abs_error"] for r in rows]

    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.yscale("log")
    plt.ylabel("Absolute error (Hartree)")
    plt.title("Energy Accuracy at 0.735 Angstrom (UCCSD)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = "fixed_abs_error_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_fixed_iterations_plot(rows):
    labels = [combo_label(r) for r in rows]
    values = [r["iterations"] for r in rows]

    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.ylabel("Optimizer iterations")
    plt.title("Convergence Iterations at 0.735 Angstrom (UCCSD)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = "fixed_iterations_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_fixed_runtime_plot(rows):
    labels = [combo_label(r) for r in rows]
    values = [r["elapsed_s"] for r in rows]

    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.ylabel("Runtime (s)")
    plt.title("Runtime at 0.735 Angstrom (UCCSD)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = "fixed_runtime_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_fixed_energy_plot(rows):
    labels = [combo_label(r) for r in rows]
    vqe_values = [r["vqe"] for r in rows]
    exact_values = [r["exact"] for r in rows]

    x = range(len(labels))
    plt.figure(figsize=(8, 4.8))
    plt.plot(x, exact_values, "k--", marker="o", label="Exact")
    plt.plot(x, vqe_values, marker="o", label="VQE")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Total energy (Hartree)")
    plt.title("Exact vs VQE Energy at 0.735 Angstrom (UCCSD)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = "fixed_total_energy_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_appendix_distance_curve(rows):
    rows = sorted(rows, key=lambda r: r["distance"])
    x = [r["distance"] for r in rows]
    exact = [r["exact"] for r in rows]
    vqe = [r["vqe"] for r in rows]

    plt.figure(figsize=(7.6, 4.6))
    plt.plot(x, exact, "k--", marker="o", label="Exact")
    plt.plot(x, vqe, marker="o", label="VQE (Jordan-Wigner + SLSQP)")
    plt.xlabel("H-H distance (Angstrom)")
    plt.ylabel("Total energy (Hartree)")
    plt.title("Supplementary Distance Validation Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = "appendix_distance_validation_curve.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main():
    fixed_rows = load_rows(FIXED_CSV)
    appendix_rows = load_rows(APPENDIX_CSV)

    fixed_rows = sorted(fixed_rows, key=lambda r: (r["mapper"], r["optimizer"]))

    outputs = [
        save_fixed_energy_plot(fixed_rows),
        save_fixed_error_plot(fixed_rows),
        save_fixed_iterations_plot(fixed_rows),
        save_fixed_runtime_plot(fixed_rows),
        save_appendix_distance_curve(appendix_rows),
    ]

    print("Generated plots:")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
