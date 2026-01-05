# Topographic Grid Cells

RNN path integrator that develops grid cell representations with topographic organization via [topoloss](https://github.com/NavneetPra/topoloss).

## Inspired By
- [DeepMind Grid Cells](https://github.com/google-deepmind/grid-cells)
- [Ganguli Lab Grid Pattern Formation](https://github.com/ganguli-lab/grid-pattern-formation)
- [Murty Lab Topoloss](https://github.com/murtylab/topoloss)

## Setup

```bash
conda env create -f environment.yml
conda activate topogrid
```

## Training

```bash
# Train without topographic loss
python train_grid.py --steps 100000

# Train with topographic loss
python train_grid.py --topoloss --steps 100000
```
