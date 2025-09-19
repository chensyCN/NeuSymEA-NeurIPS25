

# Official Code of NeurIPS'25 paper  [NeuSymEA: Neuro-Symbolic Entity Alignment via Variational Inference](https://arxiv.org/abs/2410.04153)

# QUICK START

## 1. Install the required packages

```bash
pip install -r requirements.txt
```

## 2. Run the baseline model (e.g. lightea)

```bash
python run-baseline.py --dataset fr_en --train_ratio 0.01 --ea_model lightea --gpu 1 
```

## 3. Run neuSymEA with the lightea model as base ea model

```bash
python run-neusymea.py --dataset fr_en --train_ratio 0.01 --base_ea_model lightea --gpu 1
```

## 4. Run Paris

```bash
python run-paris.py --dataset fr_en --train_ratio 0.01
```

## 5. Generate interpretations for the inferred pairs using the explainer.

```bash
python explain.py
```

### Acknowledgement

The code is based on [PRASE](https://github.com/qizhyuan/PRASE-Python), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), and [LightEA](https://github.com/MaoXinn/LightEA), the dataset is from [OpenEA benchmark](https://github.com/nju-websoft/OpenEA).

This project is licensed under the GNU General Public License v3.0 ([LICENSE](LICENSE.txt)).
