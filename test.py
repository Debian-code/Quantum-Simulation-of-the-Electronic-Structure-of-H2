from solver import Energy_calculator

def main():

    # H2 avec UCCSD + Hartree-Fock + MP2 initial point
    calc = Energy_calculator(
        atom1="H",
        atom2="H",
        distance=0.735,
        basis="sto3g",
        initial_state="hf",
        ansatz="uccsd",
        initial_point="mp2",   # tu peux tester "none" ou "hf"
    )

    # Résultat complet
    result = calc.solve()
    print(result)

    # Courbe de convergence VQE
    calc.plot_energy_curve(use_total_energy=True)


if __name__ == "__main__":
    main()
