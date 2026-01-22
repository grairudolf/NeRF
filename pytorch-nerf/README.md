# PyTorch NeRF

This project implements Neural Radiance Fields (NeRF) from scratch using PyTorch, following the original 2020 paper. The goal is to reconstruct photorealistic 3D scenes from multiple 2D images via neural volumetric rendering and ray-based sampling, mirroring a classical ray tracing pipeline.

## Overview

NeRF utilizes an implicit neural representation to map 3D coordinates `(x, y, z)` and view direction to RGB color and density. The model is trained using a photometric MSE loss, enabling it to synthesize novel views of a scene from a sparse set of input images.

## Features

- **Implicit Neural Representation**: Maps `(x, y, z, view direction)` to `(RGB, density)`.
- **Positional Encoding**: Applies Fourier encoding to enhance the model's ability to learn high-frequency details.
- **Differentiable Rendering**: Implements alpha compositing along rays for realistic rendering.
- **Stratified and Hierarchical Sampling**: Efficiently samples points along rays to improve rendering quality.
- **Novel View Synthesis**: Generates new views of the scene based on learned representations.

## Project Structure

```
pytorch-nerf
├── src
│   └── nerf
│       ├── __init__.py
│       ├── datasets.py
│       ├── dataloaders.py
│       ├── models.py
│       ├── positional_encoding.py
│       ├── renderer.py
│       ├── sampling.py
│       ├── training.py
│       ├── losses.py
│       └── utils.py
├── configs
│   └── default.yaml
├── scripts
│   ├── train.py
│   └── render.py
├── notebooks
│   └── exploration.ipynb
├── tests
│   ├── test_positional_encoding.py
│   ├── test_renderer.py
│   └── test_sampling.py
├── experiments
│   └── README.md
├── docs
│   └── paper_reference.md
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the NeRF model, execute the following command:

```
python scripts/train.py --config configs/default.yaml
```

### Rendering Novel Views

After training, you can render novel views using:

```
python scripts/render.py --config configs/default.yaml
```

## Experiments

For details on various experiments conducted with the NeRF model, refer to the `experiments/README.md` file.

## References

- Original NeRF Paper: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.