# Core Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

# Circuits / ansatz
from qiskit.circuit.library import EvolvedOperatorAnsatz

# Qiskit Nature (second_q)
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
import matplotlib.pyplot as plt
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.circuit.library import (
    HartreeFock, UCCSD, UCC, PUCCSD, PUCCD, SlaterDeterminant
)
import numpy as np


# ---------------------------------------------------------------------------
# V1 -> V2 result wrappers
# ---------------------------------------------------------------------------

class _DataBin:
    """Holds evs as a numpy array so VQE's `if not values.shape` check works."""
    def __init__(self, evs):
        self.evs = np.atleast_1d(np.real(evs))


class _V2PubResult:
    """Mimics a V2 PubResult: result.data.evs"""
    def __init__(self, value):
        self.data = _DataBin(value)
        self.metadata = {}  # observables_evaluator.py accesses .metadata


class _V2JobWrapper_Multi:
    """
    Holds pre-grouped values (one entry per original PUB).
    job.result()[i].data.evs works for both scalar and array evs.
    """
    def __init__(self, grouped_values):
        self._grouped = grouped_values

    def result(self):
        return self

    def __getitem__(self, idx):
        return _V2PubResult(self._grouped[idx])

    def __len__(self):
        return len(self._grouped)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class EstimatorV2Adapter:
    """
    Wraps a Qiskit V1 Estimator so that qiskit-algorithms VQE (which expects
    a V2 Estimator) can call it transparently.

    V2 VQE calls:  estimator.run([(circuit, observable, params), ...])
    V1 Estimator:  estimator.run(circuits, observables, parameter_values)

    The tricky part: for aux_ops evaluation, `observable` inside a PUB is a
    *list* of SparsePauliOps.  We must expand each such PUB into N individual
    (circuit, single_op, params) calls to the V1 estimator.
    """

    def __init__(self, est_v1):
        self._est = est_v1

    @staticmethod
    def _to_flat_list(params):
        """Convert any parameter representation to a plain list of Python floats."""
        if isinstance(params, np.ndarray):
            return params.flatten().tolist()
        elif hasattr(params, '__iter__'):
            return [float(p) for p in params]
        else:
            return [float(params)]

    def run(self, pubs):
        circuits         = []
        observables      = []
        parameter_values = []
        pub_sizes        = []

        for pub in pubs:
            circ   = pub[0]
            obs    = pub[1]
            params = self._to_flat_list(pub[2])

            obs_list = obs if isinstance(obs, list) else [obs]
            pub_sizes.append(len(obs_list))

            for single_obs in obs_list:
                circuits.append(circ)
                observables.append(single_obs)
                parameter_values.append(params)

        v1_job     = self._est.run(
            circuits=circuits,
            observables=observables,
            parameter_values=parameter_values,
        )
        all_values = v1_job.result().values  # numpy array, shape (total_circuits,)

        # Re-group results by original PUB
        grouped_values = []
        idx = 0
        for size in pub_sizes:
            chunk = all_values[idx: idx + size]
            grouped_values.append(chunk[0] if size == 1 else chunk)
            idx += size

        return _V2JobWrapper_Multi(grouped_values)


# ---------------------------------------------------------------------------
# Default estimator instance
# ---------------------------------------------------------------------------

def _build_default_estimator():
    from qiskit.primitives import Estimator
    return EstimatorV2Adapter(Estimator())


_DEFAULT_ESTIMATOR = _build_default_estimator()


# ---------------------------------------------------------------------------
# Energy calculator
# ---------------------------------------------------------------------------

class Energy_calculator:
    def __init__(
        self,
        atom1="H",
        atom2="H",
        distance=0.735,
        basis="sto3g",
        charge=0,
        spin=0,
        mapper=None,
        optimizer=None,
        estimator=None,
        maxiter=200,
        initial_state="hf",    # "hf" | "none" | "slater"
        ansatz="uccsd",        # "uccsd" | "ucc" | "puccsd" | "puccd"
        initial_point="mp2",   # "none" | "hf" | "mp2"
    ):
        # 1) Problem
        geom = f"{atom1} 0 0 0; {atom2} 0 0 {distance}"
        self.driver = PySCFDriver(
            atom=geom, basis=basis, charge=charge, spin=spin, unit=DistanceUnit.ANGSTROM
        )
        self.problem = self.driver.run()

        # 2) Mapper
        self.mapper = mapper if mapper is not None else JordanWignerMapper()

        # 3) Initial state circuit
        self.initial_state_name = initial_state.lower().strip()
        if self.initial_state_name == "hf":
            self.initial_state_circuit = HartreeFock(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper,
            )
        elif self.initial_state_name in ("none", "zero"):
            self.initial_state_circuit = None
        elif self.initial_state_name == "slater":
            hf_ref = HartreeFock(
                self.problem.num_spatial_orbitals,
                self.problem.num_particles,
                self.mapper,
            )
            bitstring = hf_ref._bitstr
            self.initial_state_circuit = SlaterDeterminant(
                self.problem.num_spatial_orbitals,
                bitstring,
                self.mapper,
            )
        else:
            raise ValueError("initial_state must be one of: 'hf', 'none', 'slater'")

        # 4) Ansatz
        self.ansatz_name = ansatz.lower().strip()
        ansatz_kwargs = dict(
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            qubit_mapper=self.mapper,
            initial_state=self.initial_state_circuit,
        )

        if self.ansatz_name == "uccsd":
            self.ansatz = UCCSD(**ansatz_kwargs)
        elif self.ansatz_name == "ucc":
            self.ansatz = UCC(**ansatz_kwargs)
        elif self.ansatz_name == "puccsd":
            self.ansatz = PUCCSD(**ansatz_kwargs)
        elif self.ansatz_name == "puccd":
            self.ansatz = PUCCD(**ansatz_kwargs)
        else:
            raise ValueError("ansatz must be one of: 'uccsd', 'ucc', 'puccsd', 'puccd'")

        # 5) Optimizer + Estimator
        self.optimizer = optimizer if optimizer is not None else SLSQP(maxiter=maxiter)
        self.estimator = estimator if estimator is not None else _DEFAULT_ESTIMATOR

        # 6) Initial point for VQE parameters
        self.initial_point_name = initial_point.lower().strip()
        vqe_initial_point = None

        if self.initial_point_name == "none":
            vqe_initial_point = None
        elif self.initial_point_name == "hf":
            from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
            ip = HFInitialPoint()
            ip.problem = self.problem
            ip.ansatz  = self.ansatz
            vqe_initial_point = ip.to_numpy_array()
        elif self.initial_point_name == "mp2":
            from qiskit_nature.second_q.algorithms.initial_points import MP2InitialPoint
            ip = MP2InitialPoint()
            ip.problem = self.problem
            ip.ansatz  = self.ansatz
            vqe_initial_point = ip.to_numpy_array()
        else:
            raise ValueError("initial_point must be one of: 'none', 'hf', 'mp2'")

        # 7) VQE + GroundStateEigensolver
        self.vqe = VQE(
            self.estimator,
            self.ansatz,
            optimizer=self.optimizer,
            initial_point=vqe_initial_point,
        )
        self.solver = GroundStateEigensolver(self.mapper, self.vqe)

    def solve(self):
        return self.solver.solve(self.problem)

    def plot_energy_curve(self, use_total_energy: bool = True):
        """
        Plot VQE energy at each iteration alongside the exact reference energy
        (NumPyMinimumEigensolver).

        use_total_energy=True  -> adds nuclear repulsion (total molecular energy)
        use_total_energy=False -> electronic energy only
        """

        # 1) Qubit Hamiltonian
        fermionic_hamiltonian = self.problem.hamiltonian.second_q_op()
        qubit_hamiltonian     = self.mapper.map(fermionic_hamiltonian)
        e_nuc = float(getattr(self.problem, "nuclear_repulsion_energy", 0.0))

        # 2) Exact reference with NumPy solver
        exact_solver = NumPyMinimumEigensolver()
        exact_res    = exact_solver.compute_minimum_eigenvalue(qubit_hamiltonian)
        exact_elec   = float(exact_res.eigenvalue.real)
        exact_total  = exact_elec + e_nuc
        exact_ref    = exact_total if use_total_energy else exact_elec

        # 3) Collect VQE energies via callback
        energies = []

        def cb(*args):
            # Standard signature: (eval_count, parameters, mean, std)
            if len(args) >= 3:
                mean = args[2]
                energies.append(float(getattr(mean, "real", mean)))

        # 4) Re-create VQE with callback
        initial_point = getattr(self.vqe, "initial_point", None)
        vqe_with_cb = VQE(
            self.estimator,
            self.ansatz,
            optimizer=self.optimizer,
            initial_point=initial_point,
            callback=cb,
        )
        solver_with_cb = GroundStateEigensolver(self.mapper, vqe_with_cb)
        _ = solver_with_cb.solve(self.problem)

        # 5) Convert to total energy if requested
        y = [(e + e_nuc) for e in energies] if use_total_energy else energies

        # 6) Plot
        plt.figure()
        plt.plot(range(1, len(y) + 1), y, marker="o", label="VQE")
        plt.axhline(exact_ref, linestyle="--", color="red", label="Exact")
        plt.xlabel("Iteration")
        plt.ylabel("Energy (Hartree)")
        title = ("VQE energy convergence (total)"
                 if use_total_energy else "VQE energy convergence (electronic)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig("energy.png", dpi=300)
        plt.show()

        return {
            "vqe_energies":      y,
            "exact_energy":      exact_ref,
            "exact_electronic":  exact_elec,
            "exact_total":       exact_total,
            "nuclear_repulsion": e_nuc,
        }