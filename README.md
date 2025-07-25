# logpx_samplers

Lightweight collection of diffusion model samplers.

## Overview

This repository is organized around two core components:  
- **Backbone**: wraps different diffusion pipelines behind a unified API  
- **Solver**: implements numerical sampling algorithms  

---

## Backbone

Located in `backbones/`. Each backbone wraps a specific diffusion model (e.g., HuggingFace's pipelines) and exposes a unified interface:

- `Backbone`: abstract base class with `encode()` and `sample()` methods
- Implementations:
  - `sana.py`: wraps HuggingFace `SanaPipeline`
  - `pixart_sigma.py`: wraps PixArt-Sigma
  - `dit.py`: DiT wrapper
  - `gmdit.py`: Guided-mask DiT wrapper

```
backbones/
├── backbone.py
├── sana.py
├── pixart_sigma.py
├── dit.py
└── gmdit.py
```

---

## Solvers

Located in `solvers/`. Each solver implements a specific numerical algorithm for reverse diffusion.

- `solver.py`: abstract base class
- `common.py`: shared scheduling & noise helpers
- `euler_solver.py`: Euler integrator
- `dpm_solver.py`: DPM-Solver multistep methods

```
solvers/
├── common.py
├── solver.py
├── euler_solver.py
└── dpm_solver.py
```

---

## Example Notebooks

The `examples/` directory contains end-to-end notebooks for each backbone:

```
examples/
├── sana.ipynb
├── pixart_sigma.ipynb
├── dit.ipynb
└── gmdit.ipynb
```

Each notebook demonstrates:
- Prompt-based image generation
- Backbone + Solver integration
- Sampling visualization

---

## Run Scripts

You can also generate samples in batch via:

- `runs/sample.py`: main sampling script
- `sh/run_sana.sh`: shell script for multi-CFG batch sampling with SANA

Example:

```bash
bash sh/run_sana.sh
```

Modify `sh/run_sana.sh` to change solvers, steps, CFG, etc.

---

## Installation

```bash
git clone https://github.com/yourname/logpx_samplers.git
cd logpx_samplers
pip install -r requirements.txt
```

---

## Acknowledgements

Built upon HuggingFace `diffusers`, OpenAI `guided-diffusion`, and popular solvers like DPM-Solver and UniPC.

---

## To-Do

1. AMED-Solver
2. Diff-Solver
3. GITS
4. DDP run
5. FID evaluation
6. CLIP Score evaluation
