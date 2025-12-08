# Hückel MO Solver

A Python-based Hückel Molecular Orbital (HMO) solver for pi-conjugated systems, including pure carbon systems and heteroatom-containing molecules. This solver calculates MO energies, pi electron densities, bond orders, and provides frontier orbital analysis with basic reactivity predictions.

## Features

* **Pure Carbon Systems**: Works with molecules like ethylene, butadiene, benzene, naphthalene
* **Heteroatom Support**: Handles N, O, S atoms with preset alpha/beta values
* **Eigenvalue Analysis**: Computes Hückel MO energies and eigenvectors
* **Pi Electron Densities**: Calculates per-atom electron density
* **Bond Orders**: Computes pi bond orders between bonded atoms
* **Frontier Orbitals**: Identifies HOMO, LUMO, SOMO, and predicts reactive sites
* **Custom Molecules**: Enter your own adjacency matrix and pi electron count
* **Menu-Driven Interface**: Interactive selection of molecules for analysis

## Installation

Requires Python 3.8+ and NumPy

```bash
pip install numpy
```

Clone the repository:

```bash
git clone https://github.com/yourusername/huckel-mo-solver.git
cd huckel-mo-solver
```

## Usage

Run the main script:

```bash
python main.py
```

You'll see a menu:

1. Pure carbon pi-systems
2. Heteroatom pi-systems
3. Exit

Select the type of molecule, then choose from example molecules or enter a custom adjacency matrix.

### Example: Benzene

* C6H6 cyclic system
* Adjacency matrix input (optional)
* Compute:

  * MO energies
  * Atomic coefficients
  * Pi electron densities
  * Pi bond orders
  * Frontier orbital analysis

## Heteroatom Molecules

Supported heteroatoms:

* C  - Carbon
* Np - Pyridine-type N (lone pair orthogonal to ring)
* Nr - Pyrrole-type N (lone pair contributes to ring)
* O  - Furan-type O (lone pair contributes)
* S  - Thiophene-type S (lone pair contributes)

Custom heteroatom molecules can be entered with adjacency matrix and atom types.

## Theory

* Uses Hückel Molecular Orbital Theory
* MO energies expressed as: `E = alpha + x*beta`, where alpha is Coulomb integral, beta is resonance integral, and x is eigenvalue of adjacency matrix
* Frontier orbitals (HOMO, LUMO, SOMO) are identified automatically
* Pi electron densities and bond orders computed from MO coefficients

## Limitations

* Reactivity predictions are qualitative; accurate for small conjugated systems only
* Larger systems may require more optimized implementations
* Assumes planar pi-conjugated systems

## Contributing

Contributions are welcome. Submit pull requests or report issues on GitHub.

## License

MIT License

