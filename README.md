# sergio_rs

The SERGIO v2 simulator, rewritten in Rust (approx. 150x faster than v2 upstream).

## SERGIO

SERGIO v2 (Single-cell ExpRession of Genes In silicO) is a simulator developed by Payam Dibaeinia at Saurabh Sinha’s Lab, University of Illinois at Urbana-Champaign [Sinha Lab](https://www.sinhalab.net/sinha-s-home).

## sergio_rs

sergio_rs is a reimplementation of it in Rust. I do not claim any of the actual simulator details to be original work: I simply translated the Python code, with guidance from the SERGIO paper (Dibaeinia, P. and Sinha, S., 2020. SERGIO: a single-cell expression simulator guided by gene regulatory networks. Cell systems, 11(3), pp.252-271.).

The library is exposed through a Python API using PyO3.

### Current state of the project

Currently, only manually creating GRNs is supported. To create GRNs from CSV files and other data, the user has to write manual python code to create the corresponding `Gene` objects and add them to the GRN using `add_interaction`. Additionally, only random MR initialisation is supported at the moment.

Besides that, it more or less achieves feature parity with SERGIO v2, including adding technical noise to the generated gene expression.

It is not a drop-in replacement for SERGIO v2 as I took a few liberties when designing the `sergio_rs` API, but it's very similar. For example, I have added the ability to add technical noise based on a "dataset profile" from the SERGIO paper (DS1-DS13) so you don't have to dig out the Supplementary Material to get the noise settings.

Additionally, I have added some utility functions to perform perturbations on the network. Currently, only KO perturbations are supported (which are modeled by simply making a copy of the GRN, removing the specified gene and recomputing the gene hierarchy within the graph).

## Getting Started

Install sergio_rs from PyPI using:

`pip install sergio-rs`

## Usage

### Step 1: Build a GRN

Let's assume that you already have code that parses the GRN structure from your data source. You probably have some sort of iterable that iterates over `(regulator, target, weight)` tuples.

You can create a GRN from that as follows:

```py
import sergio_rs

grn = sergio_rs.GRN()

# NOTE: you could have these vary on a gene-by-gene basis
decay = 0.8  # decay rate for the genes
n = 2  #  non-linearity of the hill function

for regulator, target, weight in my_data_iterable:
    reg = sergio_rs.Gene(name=regulator, decay=decay)
    tar = sergio_rs.Gene(name=target, decay=decay)
    grn.add_interaction(reg=reg, tar=tar, k=weight, h=None, n=n)

grn.set_mrs()
```

### Step 2: Build MR profile

**_NOTE_**: Currently, only random MR profile generation is supported.

```py
NUM_CELL_TYPES = 10
LOW_RANGE = (0, 2)
HIGH_RANGE = (2, 4)
SEED = 42

mr_profile = sergio_rs.MrProfile.from_random(
    grn,
    num_cell_types=NUM_CELL_TYPES,
    low_range=LOW_RANGE,
    high_range=HIGH_RANGE,
    seed=SEED
)
```

### Step 3: Simulate

**_NOTE_**: Resulting gene expression data are in a Polars DataFrame rather than a Pandas one. This is mostly the same as SERGIO output, with the only difference being that Polars has no "index", so the "index" with gene names is just another column in the DataFrame called "Genes".

```py
NUM_CELLS = 200
NOISE_S = 1
SAFETY_ITER = 150
SCALE_ITER = 10
DT = 0.01
SEED = 42

sim = sergio_rs.Sim(
    grn,
    num_cells=NUM_CELLS,
    noise_s=NOISE_S,
    safety_iter=SAFETY_ITER,
    scale_iter=SCALE_ITER,
    dt=DT,
    seed=SEED,
)
data = sim.simulate(mr_profile)

# Convert to 2D NumPy array
data_np = data.drop("Genes").to_numpy()
```

### Step 4: Add technical noise

```py
SEED = 42
NOISE_PROFILE = sergio_rs.NoiseSetting.DS6

noisy_data = sergio_rs.add_technical_noise(data_np, NOISE_PROFILE, seed=SEED)
```

### Step 5: Perturb

```py
GENE_TO_PERTURB = "GENE0001"
perturbed_grn, perturbed_mr_profile = grn.ko_perturbation(gene_name=GENE_TO_PERTURB, mr_profile=mr_profile)

perturbed_sim = sergio_rs.Sim(
    perturbed_grn,
    num_cells=NUM_CELLS,
    noise_s=NOISE_S,
    safety_iter=SAFETY_ITER,
    scale_iter=SCALE_ITER,
    dt=DT,
    seed=SEED,
)
perturbed_data = sim.simulate(perturbed_mr_profile)
```
