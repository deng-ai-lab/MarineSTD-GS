# gsplat for MarineSTD-GS

This directory contains the `gsplat 1.4.0` dependency used by MarineSTD-GS.

The MarineSTD-GS changes are mainly in:

- `gsplat/strategy/default.py`

The refinement changes were developed for MarineSTD-GS with reference to prior
underwater and opacity-aware Gaussian splatting methods, including
WaterSplatting and Gaussian Opacity Fields.

## Installation

Install this version from source:

```bash
pip uninstall -y gsplat
cd third_party/gsplat
pip install -e .
```

Please avoid mixing this local copy with another pip-installed version of `gsplat` in the same environment.

## Upstream Project

Upstream repository:

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
