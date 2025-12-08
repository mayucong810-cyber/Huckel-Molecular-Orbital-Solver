import numpy as np


def huckel_solver(adjacency_matrix):
    
    A = np.array(adjacency_matrix, dtype=float)
    
    if not np.allclose(A, A.T):
        raise ValueError("Adjacency matrix must be symmetric")
    
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'n_orbitals': len(eigenvalues)
    }


def format_energy(x):
    if np.isclose(x, 0):
        return "α"
    elif np.isclose(x, 1):
        return "α + β"
    elif np.isclose(x, -1):
        return "α - β"
    elif x > 0:
        return f"α + {x:.4f}β"
    else:
        return f"α - {abs(x):.4f}β"


def print_results(results):
    print("=" * 50)
    print("HUCKEL MOLECULAR ORBITAL CALCULATION RESULTS")
    print("=" * 50)
    print(f"\nNumber of π orbitals: {results['n_orbitals']}")
    
    print("\n" + "-" * 50)
    print("ENERGY LEVELS (from lowest to highest energy)")
    print("-" * 50)
    print()
    
    for i, x in enumerate(reversed(results['eigenvalues'])):
        orbital_num = results['n_orbitals'] - i
        energy_str = format_energy(x)
        print(f"  MO {orbital_num}: E = {energy_str}")
        print()
    
    print("="*50)
    print("ATOMIC COEFFICIENTS FOR LC OF MOLECULAR ORBITALS")
    print("="*50)

    n = results['n_orbitals']
    eigs = results['eigenvalues']
    vecs = results['eigenvectors']

    for mo in range(n-1, -1, -1):
        print(f"\nMO {n-mo} (eigenvalue = {eigs[mo]:+.4f})")
        print("Atom | Coefficient")
        print("--------------------")
        for atom in range(n):
            print(f"{atom+1:4d} | {vecs[atom, mo]:+.4f}")

def pi_Edensity(eigenvectors, electrons_per_mo, n):
    density = np.zeros(n)
    for k, n_elec in enumerate(electrons_per_mo):
        density += n_elec * (eigenvectors[:, k] ** 2)

    return density

def return_density(results, n_pi):
    n = results['eigenvectors'].shape[0]

    # Fill electrons in lowest MOs
    electrons_per_mo = [2] * (n_pi // 2)
    if n_pi % 2:
        electrons_per_mo.append(1)

    # Compute density
    density_result = pi_Edensity(results['eigenvectors'], electrons_per_mo, n)

    # Print nicely
    print("=" * 50)
    print("ELECTRON DENSITY ON EACH ATOM")
    print("=" * 50)
    for i, d in enumerate(density_result):
        print(f"Atom {i+1}: {d:.4f}")
    
def pi_bond_orders(eigenvectors, electrons_per_mo, matrix):
    n = len(matrix)
    bond_orders = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:     # Only compute for bonded atoms
                total = 0.0
                for k, nk in enumerate(electrons_per_mo):
                    total += nk * eigenvectors[i, k] * eigenvectors[j, k]
                bond_orders[i, j] = total

    return bond_orders

def print_bond_orders(bond_orders, matrix):
    print("=" * 50)
    print("PI BOND ORDERS")
    print("=" * 50)

    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] == 1:
                print(f"Bond {i+1}-{j+1}: {bond_orders[i,j]:.4f}")
            
def example_benzene():
    print("\n>>> EXAMPLE: BENZENE (C6H6)")
    print("    Cyclic conjugated system with 6 carbon atoms")
    
    benzene = [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0]
    ]
    n_pi = 6
    
    results = huckel_solver(benzene)
    print_results(results)
    return_density(results, n_pi)
    
    electrons_per_mo = [2] * (n_pi // 2)
    print_bond_orders(pi_bond_orders(results['eigenvectors'], electrons_per_mo, benzene), benzene)
    
    density = pi_Edensity(results['eigenvectors'], electrons_per_mo, len(benzene))
    frontier_data = frontier_analysis(results['eigenvalues'], results['eigenvectors'], n_pi, density)
    print_frontier_analysis(frontier_data, results['eigenvalues'])


def example_butadiene():
    print("\n>>> EXAMPLE: 1,3-BUTADIENE (C4H6)")
    print("    Linear conjugated system: C=C-C=C")
    
    butadiene = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    n_pi = 4
    
    results = huckel_solver(butadiene)
    print_results(results)
    return_density(results, n_pi)
    
    electrons_per_mo = [2] * (n_pi // 2)
    print_bond_orders(pi_bond_orders(results['eigenvectors'], electrons_per_mo, butadiene), butadiene)
    
    density = pi_Edensity(results['eigenvectors'], electrons_per_mo, len(butadiene))
    frontier_data = frontier_analysis(results['eigenvalues'], results['eigenvectors'], n_pi, density)
    print_frontier_analysis(frontier_data, results['eigenvalues'])


def example_ethylene():
    print("\n>>> EXAMPLE: ETHYLENE (C2H4)")
    print("    Simple double bond: C=C")
    
    ethylene = [
        [0, 1],
        [1, 0]
    ]
    n_pi = 2
    
    results = huckel_solver(ethylene)
    print_results(results)
    return_density(results, n_pi)
    
    electrons_per_mo = [2] * (n_pi // 2)
    print_bond_orders(pi_bond_orders(results['eigenvectors'], electrons_per_mo, ethylene), ethylene)
    
    density = pi_Edensity(results['eigenvectors'], electrons_per_mo, len(ethylene))
    frontier_data = frontier_analysis(results['eigenvalues'], results['eigenvectors'], n_pi, density)
    print_frontier_analysis(frontier_data, results['eigenvalues'])


def example_naphthalene():
    print("\n>>> EXAMPLE: NAPHTHALENE (C10H8)")
    print("    Two fused benzene rings")
    
    naphthalene = [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ]
    n_pi = 10
    
    results = huckel_solver(naphthalene)
    print_results(results)
    return_density(results, n_pi)
    
    electrons_per_mo = [2] * (n_pi // 2)
    print_bond_orders(pi_bond_orders(results['eigenvectors'], electrons_per_mo, naphthalene), naphthalene)
    
    density = pi_Edensity(results['eigenvectors'], electrons_per_mo, len(naphthalene))
    frontier_data = frontier_analysis(results['eigenvalues'], results['eigenvectors'], n_pi, density)
    print_frontier_analysis(frontier_data, results['eigenvalues'])


def custom_molecule():
    print("\n>>> CUSTOM MOLECULE INPUT")
    print("Enter your adjacency matrix (symmetric, 0s on diagonal), and the number of π electrons.")
    
    try:
        n = int(input("Number of atoms: "))
        print(f"Enter {n} rows of {n} space-separated values (0 or 1):")
        
        matrix = []
        for i in range(n):
            row = list(map(int, input(f"Row {i+1}: ").split()))
            if len(row) != n:
                raise ValueError(f"Expected {n} values, got {len(row)}")
            matrix.append(row)
        
        n_pi = int(input("Enter the number of π electrons: "))
        
        results = huckel_solver(matrix)
        print_results(results)
        return_density(results, n_pi)
        
        electrons_per_mo = [2] * (n_pi // 2)
        if n_pi % 2:
            electrons_per_mo.append(1)

        bond_orders = pi_bond_orders(results['eigenvectors'], electrons_per_mo, matrix)
        print_bond_orders(bond_orders, matrix)
        
        density = pi_Edensity(results['eigenvectors'], electrons_per_mo, n)
        frontier_data = frontier_analysis(results['eigenvalues'], results['eigenvectors'], n_pi, density)
        print_frontier_analysis(frontier_data, results['eigenvalues'])
        
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInput cancelled.")

HETERO_PARAMS = {
    "C":  {"alpha": 0.0,  "beta_scale": 1.00},   # reference atom
    "Np": {"alpha": 0.5,  "beta_scale": 1.00},   # pyridine N (lone pair orthogonal)
    "Nr": {"alpha": -0.5, "beta_scale": 0.80},   # pyrrole N (lone pair contributes)
    "O":  {"alpha": -1.0, "beta_scale": 0.80},   # furan O
    "S":  {"alpha": -0.8, "beta_scale": 0.90},   # thiophene S
}

def build_hetero_H(adjacency, atom_types, beta=-1.0):
    """
    atom_types: list like ["C","C","Np","C","C"] matching atoms
    beta: base β (negative)
    """

    n = len(atom_types)
    H = np.zeros((n, n))

    # Compute α values
    alpha_vals = np.zeros(n)
    for i, atom in enumerate(atom_types):
        p = HETERO_PARAMS.get(atom)
        if p is None:
            raise ValueError(f"Unknown atom type: {atom}")

        alpha_vals[i] = p["alpha"] * beta  

    # Fill Hamiltonian
    for i in range(n):
        H[i, i] = alpha_vals[i]

        for j in range(i+1, n):
            if adjacency[i][j] == 1:

                # β_ij = β × average of scaling factors
                scale_i = HETERO_PARAMS[atom_types[i]]["beta_scale"]
                scale_j = HETERO_PARAMS[atom_types[j]]["beta_scale"]

                beta_ij = beta * ((scale_i + scale_j) / 2)

                H[i, j] = beta_ij
                H[j, i] = beta_ij

    return H

def hetero_solver(H):
    """
    H: full Hamiltonian matrix
    Returns sorted energies, eigenvectors
    """
    eigvals, eigvecs = np.linalg.eigh(H)

    # Sort in ascending order
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs

def print_hetero_results(H, adjacency, atom_types, n_pi):
    # Solve Hamiltonian
    eigvals, eigvecs = hetero_solver(H)

    # Energy levels as x values
    x_vals = energy_levels_x(eigvals)

    print("="*50)
    print("ENERGY LEVELS (α + xβ form)")
    print("="*50)
    for i, x in enumerate(x_vals):
        print(f"MO {i+1}: x = {x:.4f}")

    # MO coefficients
    print("\n" + "="*50)
    print("MOLECULAR ORBITAL COEFFICIENTS")
    print("="*50)
    n = eigvecs.shape[0]

    header = "Atom | " + " | ".join([f"MO {i+1}" for i in range(n)])
    print(header)
    print("-"*len(header))

    for i in range(n):
        row = f"{i+1:4d} | " + " | ".join([f"{eigvecs[i,j]:+.4f}" for j in range(n)])
        print(row)

    # Electron density
    print("\n" + "="*50)
    print("π ELECTRON DENSITY")
    print("="*50)
    density = hetero_pi_density(eigvecs, n_pi)
    for i, d in enumerate(density):
        print(f"Atom {i+1} ({atom_types[i]}): {d:.4f}")

    # Bond orders
    print("\n" + "="*50)
    print("BOND ORDERS")
    print("="*50)
    bond_orders = compute_bond_orders(eigvecs, adjacency, n_pi)
    for (i, j), bo in bond_orders:
        print(f"Bond {i}-{j}: {bo:.4f}")

    print("="*50)
    
    # Frontier orbital analysis and reactivity predictions
    frontier_data = frontier_analysis(x_vals, eigvecs, n_pi, density)
    print_frontier_analysis(frontier_data, x_vals, atom_types)

def extract_x_values(eigvals, beta=-1.0):
    """
    Converts raw eigenvalues (α + x β) back to x
    Since α_C = 0, the raw eigenvalue = xβ
    """
    return eigvals / beta


def energy_levels_x(eigvals, beta=-1.0):
    """
    Convert eigenvalues to x values in the E = α + xβ form.
    """
    return eigvals / beta


def compute_bond_orders(eigvecs, adjacency, n_pi):
    """
    Compute bond orders for heteroatom systems.
    Returns list of ((i+1, j+1), bond_order) tuples.
    """
    n = len(adjacency)
    
    electrons_per_mo = [2] * (n_pi // 2)
    if n_pi % 2:
        electrons_per_mo.append(1)
    
    bond_orders = []
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i][j] == 1:
                total = 0.0
                for k, nk in enumerate(electrons_per_mo):
                    total += nk * eigvecs[i, k] * eigvecs[j, k]
                bond_orders.append(((i+1, j+1), total))
    
    return bond_orders


def hetero_pi_density(eigvecs, n_pi):
    """
    Compute π electron density for heteroatom systems.
    """
    n = eigvecs.shape[0]
    
    electrons_per_mo = [2] * (n_pi // 2)
    if n_pi % 2:
        electrons_per_mo.append(1)
    
    density = np.zeros(n)
    for k, n_elec in enumerate(electrons_per_mo):
        density += n_elec * (eigvecs[:, k] ** 2)
    
    return density


def main():
    print("=" * 60)
    print("         HUCKEL MOLECULAR ORBITAL THEORY SOLVER")
    print("=" * 60)
    print()
    print("This program calculates π molecular orbital energies")
    print("using the Huckel method. Energy levels are expressed as:")
    print()
    print("    E = α + x*β")
    print()
    print("where α is the Coulomb integral, β is the resonance integral,")
    print("and x is the eigenvalue of the adjacency matrix.")
    print()
    
    while True:
            print("\n" + "="*60)
            print("      HÜCKEL MO CALCULATOR — MAIN MENU")
            print("="*60)
            print("1. Pure Carbon π-System")
            print("2. Heteroatom-Containing π-System")
            print("3. Exit")
            choice = input("Select an option (1-3): ").strip()

            if choice == "1":
                carbon_menu()
            elif choice == "2":
                hetero_menu()
            elif choice == "3":
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")

def carbon_menu():
    while True:
        print("\n" + "-" * 40)
        print("PURE CARBON π-SYSTEM MENU")
        print("-" * 40)
        print("1. Ethylene (C2H4)")
        print("2. 1,3-Butadiene (C4H6)")
        print("3. Benzene (C6H6)")
        print("4. Naphthalene (C10H8)")
        print("5. Custom molecule (enter your own matrix)")
        print("6. Exit")
        print("-" * 40)
    
        try:
            choice = input("Select option (1-6): ").strip()
    
            if choice == '1':
                example_ethylene()
            elif choice == '2':
                example_butadiene()
            elif choice == '3':
                example_benzene()
            elif choice == '4':
                example_naphthalene()
            elif choice == '5':
                custom_molecule()
            elif choice == '6':
                print("\nGoodbye!")
                break
            else:
                print("Invalid option. Please enter 1-6.")
    
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
def pyridine():
    """Pyridine: 6-membered ring with one nitrogen (Np type)"""
    print("\n>>> PYRIDINE (C5H5N)")
    print("    6-membered ring with pyridine-type nitrogen")
    print("    Atom order: N-C-C-C-C-C (ring)")
    
    adjacency = [
        [0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0]
    ]
    atom_types = ["Np", "C", "C", "C", "C", "C"]
    n_pi = 6
    
    H = build_hetero_H(adjacency, atom_types)
    print_hetero_results(H, adjacency, atom_types, n_pi)


def furan():
    """Furan: 5-membered ring with oxygen"""
    print("\n>>> FURAN (C4H4O)")
    print("    5-membered ring with furan-type oxygen")
    print("    Atom order: O-C-C-C-C (ring)")
    
    adjacency = [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ]
    atom_types = ["O", "C", "C", "C", "C"]
    n_pi = 6
    
    H = build_hetero_H(adjacency, atom_types)
    print_hetero_results(H, adjacency, atom_types, n_pi)


def pyrrole():
    """Pyrrole: 5-membered ring with nitrogen (Nr type - lone pair in ring)"""
    print("\n>>> PYRROLE (C4H5N)")
    print("    5-membered ring with pyrrole-type nitrogen")
    print("    Atom order: N-C-C-C-C (ring)")
    
    adjacency = [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ]
    atom_types = ["Nr", "C", "C", "C", "C"]
    n_pi = 6
    
    H = build_hetero_H(adjacency, atom_types)
    print_hetero_results(H, adjacency, atom_types, n_pi)


def thiophene():
    """Thiophene: 5-membered ring with sulfur"""
    print("\n>>> THIOPHENE (C4H4S)")
    print("    5-membered ring with sulfur")
    print("    Atom order: S-C-C-C-C (ring)")
    
    adjacency = [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ]
    atom_types = ["S", "C", "C", "C", "C"]
    n_pi = 6
    
    H = build_hetero_H(adjacency, atom_types)
    print_hetero_results(H, adjacency, atom_types, n_pi)


def custom_hetero_molecule():
    """Custom heteroatom molecule input"""
    print("\n>>> CUSTOM HETEROATOM MOLECULE")
    print("Available atom types:")
    print("  C  - Carbon (reference)")
    print("  Np - Pyridine-type nitrogen (lone pair orthogonal to ring)")
    print("  Nr - Pyrrole-type nitrogen (lone pair contributes to ring)")
    print("  O  - Furan-type oxygen")
    print("  S  - Thiophene-type sulfur")
    print()
    
    try:
        n = int(input("Number of atoms: "))
        
        print("Enter atom types (space-separated, e.g., 'Np C C C C C'):")
        atom_types = input().strip().split()
        if len(atom_types) != n:
            raise ValueError(f"Expected {n} atom types, got {len(atom_types)}")
        
        for atom in atom_types:
            if atom not in HETERO_PARAMS:
                raise ValueError(f"Unknown atom type: {atom}")
        
        print(f"\nEnter {n} rows of {n} space-separated values (0 or 1) for adjacency matrix:")
        adjacency = []
        for i in range(n):
            row = list(map(int, input(f"Row {i+1}: ").split()))
            if len(row) != n:
                raise ValueError(f"Expected {n} values, got {len(row)}")
            adjacency.append(row)
        
        n_pi = int(input("Enter the number of π electrons: "))
        
        H = build_hetero_H(adjacency, atom_types)
        print_hetero_results(H, adjacency, atom_types, n_pi)
        
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInput cancelled.")


def hetero_menu():
    while True:
        print("\n" + "-"*50)
        print("      HETEROATOM π-SYSTEM MENU")
        print("-"*50)
        print("1. Pyridine (C5H5N)")
        print("2. Furan (C4H4O)")
        print("3. Pyrrole (C4H5N)")
        print("4. Thiophene (C4H4S)")
        print("5. Enter a custom heteroatom molecule")
        print("6. Return to main menu")
        choice = input("Select an option (1-6): ").strip()

        if choice == "1":
            pyridine()
        elif choice == "2":
            furan()
        elif choice == "3":
            pyrrole()
        elif choice == "4":
            thiophene()
        elif choice == "5":
            custom_hetero_molecule()
        elif choice == "6":
            return
        else:
            print("Invalid choice. Please enter 1-6.")

def orbital_occupancy(e_vals, electron_count):
    """
    Determines how many electrons in each MO.
    Closed-shell: 2 electrons per orbital
    Radical: last orbital gets 1 electron (SOMO)
    Cation/anions also handled automatically
    """
    n_orbitals = len(e_vals)
    occupancy = np.zeros(n_orbitals, dtype=float)

    e_left = electron_count
    i = 0
    while e_left > 0 and i < n_orbitals:
        if e_left >= 2:
            occupancy[i] = 2
            e_left -= 2
        else:
            occupancy[i] = 1
            e_left -= 1
        i += 1

    return occupancy


def identify_frontier_orbitals(e_vals, occupancy):
    """
    Find HOMO, LUMO, SOMO automatically.
    """
    occ_indices = np.where(occupancy > 0)[0]
    unocc_indices = np.where(occupancy == 0)[0]

    SOMO = None
    if any(occupancy == 1):
        SOMO = np.where(occupancy == 1)[0][0]  # first singly occupied orbital

    if len(occ_indices) > 0:
        HOMO = occ_indices[-1]
    else:
        HOMO = None

    if len(unocc_indices) > 0:
        LUMO = unocc_indices[0]
    else:
        LUMO = None

    return HOMO, LUMO, SOMO


def reaction_predictor(e_vecs, HOMO, LUMO, SOMO, densities):
    """
    Predict reactive sites:
      • For electrophiles → atom with **largest HOMO coefficient**
      • For nucleophiles → atom with **largest LUMO coefficient**
      • For radicals → atom with **largest SOMO coefficient**
      • Charge-controlled correction: also check Mulliken densities
    """
    n_atoms = e_vecs.shape[0]
    predictions = {}

    # --- Electrophilic attack (E+) ---
    if HOMO is not None:
        homo_coeffs = e_vecs[:, HOMO] ** 2
        predictions["E+_site"] = np.argmax(homo_coeffs)
        predictions["E+_weight"] = homo_coeffs[predictions["E+_site"]]

    # --- Nucleophilic attack (Nu–) ---
    if LUMO is not None:
        lumo_coeffs = e_vecs[:, LUMO] ** 2
        predictions["Nu-_site"] = np.argmax(lumo_coeffs)
        predictions["Nu-_weight"] = lumo_coeffs[predictions["Nu-_site"]]

    # --- Radical attack ---
    if SOMO is not None:
        somo_coeffs = e_vecs[:, SOMO] ** 2
        predictions["Radical_site"] = np.argmax(somo_coeffs)
        predictions["Radical_weight"] = somo_coeffs[predictions["Radical_site"]]

    # --- Charge correction: sites with highest electron density are nucleophilic ---
    predictions["Highest_density_atom"] = int(np.argmax(densities))
    predictions["Lowest_density_atom"] = int(np.argmin(densities))

    return predictions


def frontier_analysis(e_vals, e_vecs, electron_count, densities):
    """
    Wrapper function: call once after solving the molecule.
    Returns HOMO/LUMO/SOMO + predicted reactive sites.
    """
    occupancy = orbital_occupancy(e_vals, electron_count)
    HOMO, LUMO, SOMO = identify_frontier_orbitals(e_vals, occupancy)

    preds = reaction_predictor(e_vecs, HOMO, LUMO, SOMO, densities)

    return {
        "occupancy": occupancy,
        "HOMO_index": HOMO,
        "LUMO_index": LUMO,
        "SOMO_index": SOMO,
        "reaction_sites": preds
    }


def print_frontier_analysis(frontier_data, e_vals, atom_labels=None):
    """
    Print frontier orbital analysis and reactivity predictions.
    
    Args:
        frontier_data: dict from frontier_analysis()
        e_vals: eigenvalues array
        atom_labels: optional list of atom type labels (for heteroatoms)
    """
    print("\n" + "=" * 50)
    print("FRONTIER ORBITAL ANALYSIS")
    print("=" * 50)
    
    occupancy = frontier_data["occupancy"]
    HOMO = frontier_data["HOMO_index"]
    LUMO = frontier_data["LUMO_index"]
    SOMO = frontier_data["SOMO_index"]
    
    print("\nOrbital Occupancy:")
    print("-" * 30)
    for i, occ in enumerate(occupancy):
        label = ""
        if i == HOMO and SOMO is None:
            label = " <-- HOMO"
        elif i == LUMO:
            label = " <-- LUMO"
        elif i == SOMO:
            label = " <-- SOMO"
        elif i == HOMO and SOMO is not None:
            label = " <-- HOMO"
        print(f"  MO {i+1}: {int(occ)} electrons (x = {e_vals[i]:+.4f}){label}")
    
    print("\n" + "-" * 30)
    print("Frontier Orbitals:")
    print("-" * 30)
    
    if HOMO is not None:
        print(f"  HOMO: MO {HOMO + 1} (E = α + {e_vals[HOMO]:.4f}β)")
    else:
        print("  HOMO: None (no occupied orbitals)")
    
    if LUMO is not None:
        print(f"  LUMO: MO {LUMO + 1} (E = α + {e_vals[LUMO]:.4f}β)")
    else:
        print("  LUMO: None (all orbitals occupied)")
    
    if SOMO is not None:
        print(f"  SOMO: MO {SOMO + 1} (E = α + {e_vals[SOMO]:.4f}β) -- RADICAL!")
    
    if HOMO is not None and LUMO is not None:
        gap = e_vals[LUMO] - e_vals[HOMO]
        print(f"\n  HOMO-LUMO gap: {abs(gap):.4f}β")
    
    print("\n" + "=" * 50)
    print("REACTIVITY PREDICTIONS")
    print("=" * 50)
    
    preds = frontier_data["reaction_sites"]
    
    def atom_label(idx):
        if atom_labels:
            return f"Atom {idx + 1} ({atom_labels[idx]})"
        return f"Atom {idx + 1}"
    
    print("\nFrontier Molecular Orbital (FMO) Theory Predictions:")
    print("-" * 50)
    
    if "E+_site" in preds:
        site = preds["E+_site"]
        weight = preds["E+_weight"]
        print(f"  Electrophilic attack (E+): {atom_label(site)}")
        print(f"    (HOMO coefficient² = {weight:.4f})")
    
    if "Nu-_site" in preds:
        site = preds["Nu-_site"]
        weight = preds["Nu-_weight"]
        print(f"  Nucleophilic attack (Nu-): {atom_label(site)}")
        print(f"    (LUMO coefficient² = {weight:.4f})")
    
    if "Radical_site" in preds:
        site = preds["Radical_site"]
        weight = preds["Radical_weight"]
        print(f"  Radical attack: {atom_label(site)}")
        print(f"    (SOMO coefficient² = {weight:.4f})")
    
    print("\nCharge-Controlled Reactivity:")
    print("-" * 50)
    high_density = preds["Highest_density_atom"]
    low_density = preds["Lowest_density_atom"]
    print(f"  Most electron-rich (nucleophilic): {atom_label(high_density)}")
    print(f"  Most electron-poor (electrophilic): {atom_label(low_density)}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
