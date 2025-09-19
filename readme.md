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