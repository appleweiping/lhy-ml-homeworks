# HW1 — Regression (COVID-19 case prediction)

DNN regression predicting the 4th-day tested-positive rate from a 3-day window
of US-state COVID survey features. Kaggle: `ml2022spring-hw1`.

## Run
```bash
python hw1_regression.py --epochs 200
```
Uses `data/covid.train.csv` if present, else synthesises a dataset with the
official 94-feature layout (40 state one-hots + 3×18 survey blocks).

Get real data: `kaggle competitions download -c ml2022spring-hw1 -p data && unzip data/*.zip -d data`

## Measured result (CPU, 3 threads, synthetic official-layout data)
| metric | value |
|---|---|
| valid MSE | 2.14 |
| valid RMSE | 1.46 |

Model: MLP 94→64→32→1, SGD(lr=1e-3, momentum=0.9), MSE loss, early stopping.
Outputs: `results/metrics.txt`, `results/submission.csv`.
