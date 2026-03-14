# ⚛️ Quantum Simulation of H₂ Electronic Structure using VQE
 
> A hybrid quantum-classical pipeline to approximate the ground-state energy of the hydrogen molecule using Variational Quantum Algorithms.
 
**Course:** QUANTIQUE 1 — ESILV 2025/2026  
**Authors:** Raphael Marques Araujo · Youssef Benaddi · Tony Pansera · Iris Vermeil
 
---
 
## 📋 Table of Contents
 
- [Overview](#overview)
- [Physics Background](#physics-background)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
 
---
 
## Overview
 
This project implements a complete, reproducible pipeline for **quantum simulation of molecular electronic structure**, targeting the H₂ molecule as a benchmark system.
 
The goal is to approximate the **ground-state energy** of H₂ by mapping its electronic Hamiltonian onto a qubit system and solving it with the **Variational Quantum Eigensolver (VQE)** algorithm — a hybrid quantum-classical approach well suited for near-term quantum hardware.
 
The pipeline is built using:
- **[PySCF](https://pyscf.org/)** — for molecular integral generation (Hartree–Fock, basis sets)
- **[Qiskit](https://qiskit.org/)** — for qubit encoding, ansatz construction, and VQE optimization
 
---
 
## Physics Background
 
### The Electronic Hamiltonian
 
The full molecular Hamiltonian is decomposed into kinetic and Coulomb interaction terms:
 
$$\hat{H} = \hat{T}_e + \hat{T}_N + \hat{V}_{ee} + \hat{V}_{NN} + \hat{V}_{eN}$$
 
Under the **Born–Oppenheimer approximation**, nuclear kinetic energy is neglected, reducing the problem to a purely electronic Hamiltonian. This is then reformulated in **second quantization**:
 
$$\hat{H}_{el} = \sum_{ij} h_{ij} a_i^\dagger a_j + \frac{1}{2} \sum_{ijkl} h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l$$
 
### Basis Set: STO-3G
 
Each atomic orbital is approximated as a linear combination of 3 Gaussian functions (STO-3G), enabling analytical evaluation of all molecular integrals while maintaining physical accuracy.
 
### Jordan–Wigner Mapping
 
Fermionic creation/annihilation operators are mapped to qubit (Pauli) operators via the Jordan–Wigner transformation, preserving the required anticommutation relations:
 
$$a_i = \left(\prod_{k<i} Z_k\right) \frac{X_i + iY_i}{2}$$
 
For H₂ in STO-3G, this yields a **4-qubit system** (2 spatial orbitals × 2 spin components).
 
---
 
## Methodology
 
### Pipeline Overview
 
```
Molecule (H₂)
    │
    ▼
PySCF: Hartree–Fock + STO-3G integrals
    │
    ▼
Second-quantized Hamiltonian
    │
    ▼
Fermion-to-qubit mapping (Jordan–Wigner / Parity / Bravyi–Kitaev)
    │
    ▼
VQE: Ansatz + Classical Optimizer → Ground-state energy E₀
```
 
### Ansatz Families
 
| Ansatz | Description | Avg. Iterations | Avg. Runtime |
|--------|-------------|-----------------|--------------|
| **UCCSD** | Unitary Coupled Cluster (singles + doubles) | ~49.8 | ~2.42 s |
| **PUCCSD** | Paired UCCSD — reduced complexity | intermediate | intermediate |
| **PUCCD** | Paired doubles only — most compact | ~17.2 | ~0.34 s |
 
### Fermionic Mappings Tested
 
- **Jordan–Wigner** — direct, qubit-local, no qubit reduction
- **Parity** — encodes parity information; supports 2-qubit reduction
- **Bravyi–Kitaev** — balanced locality between occupation and parity
 
### Classical Optimizers
 
- **SLSQP** — gradient-based, fast convergence
- **COBYLA** — gradient-free, more robust to noise
 
---

## Results
 
All 18 configurations were evaluated at d = 0.735 Å. Absolute errors ranged from **10⁻¹³ to 10⁻⁷ Hartree**, well within chemical accuracy.
 
| Best configuration | Error |
|---|---|
| PUCCD + Bravyi–Kitaev + SLSQP | 1.04 × 10⁻¹³ Hartree ✅ |
| PUCCSD + Bravyi–Kitaev + SLSQP | 2.97 × 10⁻⁷ Hartree (worst) |
 
### Optimizer comparison (averaged over all mappings & ansatz)
 
| Optimizer | Mean Iterations | Mean Runtime | Mean Error |
|-----------|----------------|--------------|------------|
| SLSQP | ~24.0 | ~0.96 s | higher variance |
| COBYLA | ~40.8 | ~1.56 s | lower mean error |
 
### Mapping comparison (averaged over all optimizers & ansatz)
 
| Mapping | Mean Absolute Error |
|---------|---------------------|
| Parity | ~1.47 × 10⁻⁸ Hartree (best) |
| Jordan–Wigner | ~6 × 10⁻⁸ Hartree |
| Bravyi–Kitaev | ~6 × 10⁻⁸ Hartree |
 
---
 
## Key Findings
 
- **All 18 VQE runs** converged close to the exact ground-state energy, confirming the pipeline's correctness.
- **Optimizer choice** has the strongest impact on runtime and iteration count. SLSQP is faster; COBYLA is more accurate on average.
- **Mapping choice** primarily affects accuracy. The Parity mapping achieved the lowest mean error overall.
- **PUCCD** offers the best efficiency trade-off: fewest iterations, fastest runtime, and competitive accuracy — making it the recommended ansatz for small systems like H₂.
- The VQE energies closely follow the exact dissociation curve across d ∈ {0.50, 0.735, 1.00, 1.50} Å, validating the fixed-geometry conclusions.
 
---
 
## Future Work
 
- Extend to larger active spaces (e.g., LiH, BeH₂)
- Explore adaptive ansatz strategies (ADAPT-VQE)
- Implement measurement-cost reduction via Pauli grouping
- Validate on noisy hardware backends (IBM Quantum, IonQ)
- Benchmark against classical FCI and CCSD(T) references
 
---
 
## References
 
- Peruzzo et al., *A variational eigenvalue solver on a photonic chip*, Nature Communications (2014)
- O'Malley et al., *Scalable quantum simulation of molecular energies*, Physical Review X (2016)
- Qiskit Nature documentation: https://qiskit-community.github.io/qiskit-nature/
- PySCF documentation: https://pyscf.org/
 
---
 
## License
 
This project is for academic purposes (ESILV QUANTIQUE 1 course, 2025/2026).
