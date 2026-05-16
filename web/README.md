---
title: VAE-FDM Explorer
emoji: 🏗️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
tags:
  - structural-engineering
  - form-finding
  - jax
  - three-js
  - force-density-method
short_description: Structural form-finding with differentiable mechanics
---

# VAE-FDM Explorer

3D interactive tool for structural form-finding. A neural encoder predicts force densities and a mechanical decoder (Force Density Method) produces the equilibrium shape.

Drag control points or adjust sliders to explore shell geometries. All shapes satisfy mechanical equilibrium — the FDM decoder solves it directly.

Built on top of the work by Pastrana et al., ICLR 2025.

## How it works

- **Backend**: FastAPI serving a JAX neural network (~2ms inference on CPU)
- **Frontend**: Three.js with real-time 3D interaction
- **Physics**: Force Density Method guarantees equilibrium for every prediction

## Links

- [Paper (arXiv)](https://arxiv.org/abs/2409.02606)
- [Source code](https://github.com/efradeca/vae-fdm)

## Configuration

Check the [Spaces Docker SDK reference](https://huggingface.co/docs/hub/spaces-sdks-docker) for more details.
