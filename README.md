# pytmod

<!-- start badges -->

[![tests](https://img.shields.io/github/actions/workflow/status/benvial/pytmod/test.yml?label=tests&style=for-the-badge)](https://github.com/benvial/pytmod/actions)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge&logo=google-docs&logoColor=white&logoSize=auto&labelColor=%23202020&color=%234FA3A0)](https://opensource.org/licenses/MIT)

<!-- [![PyPI version](https://img.shields.io/pypi/v/pytmod?style=for-the-badge&logo=pypi&logoColor=white&logoSize=auto&labelColor=%23202020&color=%234FA3A0)](https://pypi.org/project/pytmod)
[![Python Versions](https://img.shields.io/pypi/pyversions/pytmod?style=for-the-badge&logo=python&logoColor=white&logoSize=auto&labelColor=%23202020&color=%234FA3A0)](https://pypi.org/project/pytmod/) -->
<!-- end badges -->

<!-- start intro -->

**pytmod** is a Python library for solving electromagnetic wave scattering and quasi-normal modes (QNMs) in time-modulated media.

It provides tools to analyze how periodic temporal modulation of material properties (e.g., permittivity) affects wave propagation, scattering, and resonant mode spectra. The library supports both slab and cylinder geometries.

## Key Features

- **Floquet Framework**: Compute scattering coefficients and quasi-normal modes for slabs and cylinders with time-periodic permittivity modulation using Floquet theory.
- **Material Modeling**: Define time-modulated permittivity profiles.
- **Multiple Geometries**: Support for both 1D slab and 2D cylindrical scatterers with time-modulated material properties.
- **QNM Solver**: Find and track quasi-normal modes in the complex-frequency plane as modulation parameters vary.
- **QNM expansion**: Use QNMs to expand the scattered field.
- **Scattering Analysis**: Calculate scattering cross-sections and field distributions with sideband generation at Floquet harmonics.
- **FEM Support**: Finite Element Method solver for time-modulated slab eigenvalue problems using FEniCSx/dolfinx (optional).

<!-- end intro -->

<!-- start install -->

## Prerequisites

To use pytmod, you must have the following installed on your system:

1.  **Python**: Version 3.10 or higher.
2.  **Scientific Python Stack**: NumPy, SciPy, and Matplotlib (installed automatically with pytmod).

## Installation

Install pytmod directly from PyPI:

```bash
pip install pytmod
```

For development or to include documentation and testing dependencies:

```bash
pip install "pytmod[dev]"
```

### FEM Support (Optional)

To use the Finite Element Method (FEM) solver for time-modulated slab problems, you need to install additional dependencies. The FEM solver requires FEniCSx/dolfinx which is best installed via conda:

**Using conda (recommended):**

```bash
conda install -c conda-forge fenics-dolfinx 'petsc=*=complex*'
```

Then install the pip dependencies:

```bash
pip install "pytmod[fem]"
```

**Using the provided environment file:**

```bash
conda env create -f environment.yml
conda activate pytmod
```

Alternatively, to build from source, clone the repository and install using `pip`:

```bash
git clone git@github.com:benvial/pytmod.git
cd pytmod
pip install .
```

<!-- end install -->

## Contributing

Contributions are welcome! To set up a development environment:

1.  Clone the repository:

    ```bash
    git clone https://github.com/benvial/pytmod.git
    cd pytmod
    ```

2.  Install dependencies (using `just` or `pip`):

    ```bash
    just install-dev
    ```

3.  Run tests:
    ```bash
    just test
    ```

## Documentation

Full documentation can be found at [bvial.info/pytmod](https://bvial.info/pytmod)

## License

This project is licensed under the MIT License.
