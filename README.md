# SERGIO-JAX: Fast simulation of single-cell gene expressions
Authors: Yunshu Ouyang, Alexander Hägele

## What is SERGIO?
SERGIO is a simulator for single-cell expression data guided by gene regulatory networks (GRNs). The GRN can be a user-specified parameter, which allows for flexible simulation of data that closely ressembles real-world datasets. SERGIO can simulate any number of cell types and number of cells (samples). Mathematically, the simulator uses stochastic differential equations based on the Chemical Langevin Equation (CLE).


Please refer to the original paper for more details:
```bash
Dibaeinia, P., & Sinha, S. (2020). SERGIO: a single-cell expression simulator guided by gene regulatory networks. Cell systems, 11(3), 252-271.
```
* Link: https://www.sciencedirect.com/science/article/pii/S2405471220302878
* GitHub: https://github.com/PayamDiba/SERGIO

## What does this repository contain?
This repository provides a reimplementation of SERGIO ***in JAX***. It is `jit`-compatible, which allows the usage of hardware-accelerators (such as GPUs) for fast inference. We focus on the steady-state simulation of the cell types.

## Usage
We include a simple notebook in [run_sergio.ipynb](run_sergio.ipynb) that demonstrates the usage of the `SergioJAX` class.
The basic usage was designed to closely follow the [original implementation](https://github.com/PayamDiba/SERGIO) and can be summarized as follows:

* First, instantiate the simulator via
```python
sim = SergioJAX(
    n_genes=n_genes,
    n_cell_types=n_cell_types,
    n_sc=n_sc,
    noise_amplitude=noise_amplitude,
    noise_type=noise_type,
    decays=decays,
)
```
* Then, initialize the graph parameters
```python
sim.custom_graph(
    graph,
    k,
    basal_rates,
    hill,
)
```
* Finally, run the simulation and extract the clean data
``` python
# rng is jax random key
rng, subrng = jax.random.split(rng)
sim.simulate(rng=subrng)

# shape: [number_bins (#cell types), number_genes, number_sc (#cells per type)]
rng, subrng = jax.random.split(rng)
expr = sim.getExpressions(rng=subrng)
```

Additionally, it is possible to add technical noise to the expressions, as outlined in [run_sergio.ipynb](run_sergio.ipynb).

---
### Dependencies
SERGIO-JAX only depends on `jax` and `jaxlib`. The code was tested with version `jax` version 0.4.12 and Python 3.8.16. 

Our example notebook additionally uses `umap-learn`, `igraph`, `matplotlib`, and `scikit-learn`, but the backbone SERGIO simulator does not require these packages.


## Credits
If you use SERGIO-JAX in your project, please cite the [original paper](https://www.sciencedirect.com/science/article/pii/S2405471220302878) and acknowledge our repository!

_Yunshu Ouyang, Alexander Hägele. 2023._
