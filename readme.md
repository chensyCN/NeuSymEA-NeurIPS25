

### Code for NeurIPS'25 paper  [Neuro-Symbolic Entity Alignment via Variational Inference](https://arxiv.org/abs/2410.04153)

## Quick Start

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

### 2. Run the baseline model (e.g. lightea)

```bash
python run-baseline.py --dataset fr_en --train_ratio 0.01 --ea_model lightea --gpu 1 
```

### 3. Run neuSymEA with the lightea model as base ea model

```bash
python run-neusymea.py --dataset fr_en --train_ratio 0.01 --base_ea_model lightea --gpu 1
```

### 4. Run Paris

```bash
python run-paris.py --dataset fr_en --train_ratio 0.01
```

### 5. Generate interpretations for the inferred pairs using the explainer.

```bash
python explain.py
```

# 🍀 Citation
If you find this work helpful, please cite our paper:
```
@inproceedings{chen2025neusymea,
title={NeuSym{EA}: Neuro-symbolic Entity Alignment via Variational Inference},
author={Shengyuan Chen, Zheng Yuan, Qinggang Zhang, Wen Hua, Jiannong Cao, Xiao Huang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=SAbQLqf8XL}
}
```


## Acknowledgement

The code is based on [PRASE](https://github.com/qizhyuan/PRASE-Python), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), and [LightEA](https://github.com/MaoXinn/LightEA), the dataset is from [OpenEA benchmark](https://github.com/nju-websoft/OpenEA).

This project is licensed under the GNU General Public License v3.0 ([LICENSE](LICENSE.txt)).
