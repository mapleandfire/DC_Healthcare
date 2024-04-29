# Medical records condensation: Medical records condensation: a roadmap towards healthcare data democratisation

Code is tested under `Ubuntu==20.04` and a RTX-3060 card.

## Dependencies
* We recommend to use [Anaconda](https://www.anaconda.com/) for Python environment management. 
* Create a python environment named "DC" and activate it.
```
conda create --name dc python==3.9.12
conda activate dc
```

* Install [PyTorch](https://pytorch.org/get-started/locally/). We recommend to use CUDA for computational acceleration.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

* Install all other dependencies (einops, scikit-learn, easydict, pandas, tensorboardx, tensorboard)
```
pip install -r requirements.txt
```

## Learning Condensed Dataset 
NOTE: datasets are not included here; you should download them from websites as described in the paper, and place them as follows (pre-processing may also be needed):
```
..
├── $ROOT
│   ├── README.md
│   ├── requirements.txt
│   ...
├── datasets
│   └── mimic3
│   └── PhysioNet
│   └── covid
│       └── breath
│       ...
```

Training data like checkpoints and synthetic data will be saved under ```$ROOT/../snapshots/```. Before run the session, please make sure this path is reserved on your machine to avoid any data corruption.

* Learn a condensed dataset on [PhysioNet-2012](https://physionet.org/content/challenge-2012/1.0.0/)
```
python train_DC.py --dataset "physio" --sp "24000" --eval_sp "24000" --dm_ipc "40" --prpr "std" --save_dir_name "PhysioNet_DC_exp1"
```

* Learn a condensed dataset on [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
```
python train_DC.py --dataset "mimic3" --sp "24000" --eval_sp "24000" --dm_ipc "400" --prpr "none" --save_dir_name "MIMIC3_DC_exp1"
```

* Learn a condensed dataset on [Coswara](https://github.com/iiscleap/Coswara-Data)
```
python train_DC.py --dataset "covid_b" --sp "24000" --eval_sp "24000" --dm_ipc "40" --prpr "std" --save_dir_name "Coswara_DC_exp1"
```

We also provide Jupyter Notebooks on learning DC on three datasets: `./PhysioNet_DC.ipynb`, `./MIMIC-III_DC.ipynb` and `./Coswara_DC.ipynb`. Note that the AUCs may be slightly different from the ones in the papers, as they are separate training sessions.
