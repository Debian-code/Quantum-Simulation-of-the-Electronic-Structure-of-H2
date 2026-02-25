import csv
import time
from dataclasses import dataclass

import numpy as np

from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SLSQP, COBYLA

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD


FIXED_DISTANCE = 0.735
DISTANCE_SWEEP_APPENDIX = [0.50, 0.735, 1.00, 1.50]
ANSATZ_NAME = "uccsd"


@dataclass
class RunResult:
    distance: float
    mapper: str
    optimizer: str
    exact_total: float
    vqe_total: float
    abs_error: float
    iterations: int
    elapsed_s: float


def build_problem(distance: float, basis: str = "sto3g"):
    driver = PySCFDriver(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis=basis,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    return driver.run()


def build_mapper(name: str):
    if name == "jordan_wigner":
        return JordanWignerMapper()
    if name == "parity":
        return ParityMapper()
    if name == "bravyi_kitaev":
        return BravyiKitaevMapper()
    raise ValueError(f"Unknown mapper: {name}")


def build_optimizer(name: str):
    if name == "slsqp":
        return SLSQP(maxiter=120)
    if name == "cobyla":
        return COBYLA(maxiter=120)
    raise ValueError(f"Unknown optimizer: {name}")


def build_ansatz(problem, mapper):
    hf = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    )
    return UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        initial_state=hf,
    )


def run_single(distance: float, mapper_name: str, optimizer_name: str) -> RunResult:
    mapper = build_mapper(mapper_name)
    problem = build_problem(distance)

    fermion_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermion_op)
    e_nuc = float(getattr(problem, "nuclear_repulsion_energy", 0.0))

    exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
    exact_total = float(np.real(exact.eigenvalue)) + e_nuc

    ansatz = build_ansatz(problem, mapper)
    optimizer = build_optimizer(optimizer_name)

    history = []

    def cb(*args):
        if len(args) >= 3:
            history.append(float(np.real(args[2])))

    vqe = VQE(
        estimator=StatevectorEstimator(),
        ansatz=ansatz,
        optimizer=optimizer,
        callback=cb,
    )

    t0 = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    elapsed_s = time.perf_counter() - t0

    vqe_total = float(np.real(result.eigenvalue)) + e_nuc

    return RunResult(
        distance=distance,
        mapper=mapper_name,
        optimizer=optimizer_name,
        exact_total=exact_total,
        vqe_total=vqe_total,
        abs_error=abs(vqe_total - exact_total),
        iterations=len(history),
        elapsed_s=elapsed_s,
    )


def run_fixed_grid():
    mappers = ["jordan_wigner", "parity", "bravyi_kitaev"]
    optimizers = ["slsqp", "cobyla"]

    results = []
    total = len(mappers) * len(optimizers)
    idx = 0

    for mapper in mappers:
        for optimizer in optimizers:
            idx += 1
            print(
                f"[fixed {idx}/{total}] distance={FIXED_DISTANCE:.3f} mapper={mapper} optimizer={optimizer}",
                flush=True,
            )
            r = run_single(FIXED_DISTANCE, mapper, optimizer)
            results.append(r)
            print(
                f"  -> vqe={r.vqe_total:.8f} exact={r.exact_total:.8f} "
                f"abs_error={r.abs_error:.2e} iters={r.iterations} t={r.elapsed_s:.2f}s",
                flush=True,
            )

    with open("fixed_point_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "distance_A",
            "ansatz",
            "mapper",
            "optimizer",
            "exact_total_hartree",
            "vqe_total_hartree",
            "abs_error_hartree",
            "iterations",
            "elapsed_s",
        ])
        for r in results:
            w.writerow([
                f"{r.distance:.3f}",
                ANSATZ_NAME,
                r.mapper,
                r.optimizer,
                f"{r.exact_total:.10f}",
                f"{r.vqe_total:.10f}",
                f"{r.abs_error:.10e}",
                r.iterations,
                f"{r.elapsed_s:.4f}",
            ])

    print(f"Saved {len(results)} rows to fixed_point_results.csv")


def run_appendix_sweep():
    mapper = "jordan_wigner"
    optimizer = "slsqp"

    results = []
    total = len(DISTANCE_SWEEP_APPENDIX)

    for i, distance in enumerate(DISTANCE_SWEEP_APPENDIX, start=1):
        print(
            f"[appendix {i}/{total}] distance={distance:.3f} mapper={mapper} optimizer={optimizer}",
            flush=True,
        )
        r = run_single(distance, mapper, optimizer)
        results.append(r)
        print(
            f"  -> vqe={r.vqe_total:.8f} exact={r.exact_total:.8f} "
            f"abs_error={r.abs_error:.2e} iters={r.iterations} t={r.elapsed_s:.2f}s",
            flush=True,
        )

    with open("appendix_distance_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "distance_A",
            "ansatz",
            "mapper",
            "optimizer",
            "exact_total_hartree",
            "vqe_total_hartree",
            "abs_error_hartree",
            "iterations",
            "elapsed_s",
        ])
        for r in results:
            w.writerow([
                f"{r.distance:.3f}",
                ANSATZ_NAME,
                r.mapper,
                r.optimizer,
                f"{r.exact_total:.10f}",
                f"{r.vqe_total:.10f}",
                f"{r.abs_error:.10e}",
                r.iterations,
                f"{r.elapsed_s:.4f}",
            ])

    print(f"Saved {len(results)} rows to appendix_distance_curve.csv")


def main():
    print(f"Running fixed-point benchmark at {FIXED_DISTANCE:.3f} Angstrom with ansatz={ANSATZ_NAME}")
    run_fixed_grid()
    print("\nRunning supplementary distance sweep for appendix")
    run_appendix_sweep()


if __name__ == "__main__":
    main()
