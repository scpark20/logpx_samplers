# logpx_samplers

Lightweight collection of diffusion model samplers

## Overview

This repository is organized around two core components:  
- **Backbone**: wraps different diffusion pipelines behind a unified API  
- **Solver**: implements numerical sampling algorithms  

---

## Backbone

Located in `backbones/`. Each Backbone handles prompt encoding and denoising calls:

- **backbone.py**  
  - `Backbone` abstract base class  
  - Defines `encode()` and `sample()` interfaces, device setup  

- **sana.py**  
  - `SANA` implementation wrapping HuggingFace’s `SanaPipeline`  
  - Converts text prompts into latents and runs denoising steps  

```
backbones/
├── backbone.py      # abstract pipeline interface
└── sana.py          # SanaPipeline wrapper
```

---

## Solvers

Located in `solvers/`. Each Solver inherits from the `Solver` interface and performs step-wise sampling:

- **solver.py**  
  - `Solver` abstract base class  
  - Defines core methods like `update()` and `step()`

- **common.py**  
  - Scheduling functions, noise utilities, shared helpers  

- **euler_solver.py**  
  - Implements classic Euler sampling method  

- **dpm_solver.py**  
  - Implements DPM-Solver algorithms (higher-order multistep)  

```
solvers/
├── common.py        # schedules & utilities
├── solver.py        # Solver abstract class
├── euler_solver.py  # Euler sampler
└── dpm_solver.py    # DPM-Solver sampler
```

---

## Installation

```bash
git clone https://github.com/yourname/logpx_samplers.git
cd logpx_samplers
pip install -r requirements.txt
```
