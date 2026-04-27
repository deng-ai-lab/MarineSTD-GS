# Modified gsplat for MarineSTD-GS

This directory contains a **method-specific modified version derived from gsplat 1.4.0** for reproducing **MarineSTD-GS**.

Compared with the upstream gsplat release, this version includes method-specific modifications required by MarineSTD-GS. In particular, we modify the Gaussian refinement strategy in:

- `gsplat/strategy/default.py`

The modified refinement strategy is a method-specific adjustment used by
MarineSTD-GS. Its design was developed with reference to prior underwater and
opacity-aware Gaussian-splatting methods, including WaterSplatting and
Gaussian Opacity Fields.

This copy is provided **only for reproducing MarineSTD-GS** and should **not** be treated as the official upstream gsplat package.

## Installation

**Dependency:** Please install [PyTorch](https://pytorch.org/get-started/locally/) first.

This modified gsplat is intended to be installed **locally from source**.

From the root directory of MarineSTD-GS, run:

```bash
cd third_party/gsplat
pip install -e .
```

If an official version of `gsplat` was automatically installed together with Nerfstudio, we recommend removing it first:

```bash
pip uninstall -y gsplat
cd third_party/gsplat
pip install -e .
```

## Notes

- This is a **method-specific modified version** of **gsplat 1.4.0** used by MarineSTD-GS.
- It is **not** the official release from the gsplat authors.
- Please avoid mixing this local version with another pip-installed version of gsplat in the same environment.
- The primary purpose of this directory is to provide a reproducible dependency for MarineSTD-GS.

## Upstream Project

The original gsplat project is an open-source library for CUDA-accelerated rasterization of gaussians with Python bindings. It is inspired by the SIGGRAPH paper:

- [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

The upstream gsplat project is available at:

- [https://github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)

We gratefully acknowledge the authors and contributors of the original gsplat project.

## Original Citation

If you find the original gsplat library useful in your projects or papers, please consider citing:

```bibtex
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting},
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765}
}
```

## License

This directory is derived from the original gsplat project. Please refer to the original license files included in this directory for licensing details.
