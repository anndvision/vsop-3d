# vsop-3d

![Min-Max Scores](/assets/min-max-scores.png)

## installation

```.sh
git clone git@github.com:anndvision/vsop-3d.git
cd vsop-3d
conda env create -f environment.yml
conda activate vsop-3d
pip install [-e] .
```

## Run algorithm

```.sh
python vsop_3d/vsop_3d_procgen.py --experiment-id my-experiment --seed 0 --job-dir experiments/ --track True --wandb-entity [my-entity] --env-id starpilot
```

## Citation

Thank you for using our work. Please consider citing if you use it in your own research.

```.bib
@article{jesson2024improved,
  title={Improving Generalization on the ProcGen Benchmark with Simple Architectural Changes and Scale},
  author={Jesson, Andrew and Jiang, Yiding},
  journal={arXiv preprint arXiv:????.?????},
  year={2024}
}
```
