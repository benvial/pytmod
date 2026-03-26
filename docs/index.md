---
layout: landing
description: pytmod is a Python library for solving electromagnetic wave scattering and quasi-normal modes in time-modulated media.
---

```{image} _static/pytmod.svg
:width: 150px
:alt: pytmod
```

# pytmod

```{include} ../README.md
:start-after: <!-- start badges -->
:end-before: <!-- end badges -->
```

```{rst-class} lead
Python library for electromagnetic wave scattering and quasi-normal modes in time-modulated media.
```

```{container} buttons

[{octicon}`book;1.1em;icon-magin-right` Docs](/install)
[{octicon}`list-unordered;1.1em;icon-magin-right` Examples](/examples/index)
[{octicon}`code;1.1em;icon-magin-right` API](/autoapi/pytmod/index)
[{octicon}`mark-github;1.1em;icon-magin-right` GitHub](https://github.com/benvial/pytmod)
```

## Features

::::{grid} 1 1 2 2
:gutter: 2
:padding: 0
:class-row: surface

:::{grid-item-card} {iconify}`mdi:sine-wave` Floquet Framework
Compute scattering coefficients and quasi-normal modes for slabs and cylinders with time-periodic permittivity modulation using Floquet theory.
:::
:::{grid-item-card} {iconify}`mdi:layers-triple` Material Modeling
Define time-modulated permittivity profiles with arbitrary periodic modulation.
:::
:::{grid-item-card} {iconify}`mdi:shape` Multiple Geometries
Support for both 1D slab and 2D cylindrical scatterers with time-modulated material properties.
:::
:::{grid-item-card} {iconify}`mdi:chart-bell-curve-cumulative` QNM Solver
Find and track quasi-normal modes in the complex-frequency plane as modulation parameters vary.
:::
:::{grid-item-card} {iconify}`mdi:math-integral-box` QNM Expansion
Use quasi-normal modes to expand the scattered field and analyze resonant contributions.
:::
:::{grid-item-card} {iconify}`mdi:chart-scatter-plot-hexbin` Scattering Analysis
Calculate observables such as reflection and transmission or scattering cross-sections as well field distributions.
:::
::::

```{toctree}
:caption: Getting started
:hidden:
install
```

```{toctree}
:caption: Examples
:hidden:
examples/index.rst
```

```{toctree}
:caption: API
:hidden:
autoapi/pytmod/index
```

```{toctree}
:caption: References
:hidden:
biblio
```
