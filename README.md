# Restaurant Turnover Prediction

This project predicts annual restaurant turnover in INR from restaurant attributes, ratings and amenities. The final submission contains 500 predictions for 500 test restaurants. The training set contains 3,493 restaurants.

## Model and results

The final V14 ensemble combines 12.5% V7, 12.5% TabPFN v2, and 75% TabPFN-3.5 Base. V7 uses the verified fitted state and its training source is in `src/`. The two TabPFN components fit from the training data at reproduction time. Features exclude the target and restaurant ID. The ID aligns predictions with the test rows.

| Metric | RMSE in INR |
| --- | ---: |
| V14 recorded submission score | 11,736,110.79 |
| V9 recorded submission score | 11,904,515.99 |

V14 lowered the recorded score by 1.41%. The saved confirmation results cover five split schemes, including a held-out-city scheme. The previous fresh Tesla T4 execution matched the saved predictions within INR 0.01. See [confirmation results](validation/confirmation.csv) and [GPU reproduction record](validation/gpu_reproduction.json).

## Check the submission

Run this command from the repository root with Python 3.12. It checks the delivered input and artifact checksums and validates the CSV headers, restaurant IDs, row order and predicted values.

```bash
python src/check_submission.py
```

## Reproduce the predictions

Use a Linux environment with Python 3.12, an NVIDIA Tesla T4 and support for the CUDA 13.0 PyTorch build. The first run downloads the two pinned TabPFN checkpoints from Hugging Face and checks their SHA-256 digests. The original fresh run took about 31 minutes.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.11.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python src/reproduce.py
python src/check_submission.py outputs/submission.csv
```

Open [submission.ipynb](submission.ipynb) from the repository root for the same checks and reproduction workflow in Jupyter. The reproduction script verifies the numerical result against the scored submission before writing `outputs/submission.csv`.

## Files

| Path | Contents |
| --- | --- |
| `submission.csv` | Byte-identical copy of the scored predictions |
| `submission.ipynb` | Notebook for validation and reproduction |
| `src/` | Data preparation, feature engineering, fitting, inference and checks |
| `data/` | Training and test data, data dictionary and sample submission |
| `artifacts/` | Fitted V7 checkpoint, reference predictions and pinned checksums |
| `validation/` | Historical confirmation and GPU reproduction results |

TabPFN v2 is described in [Hollmann et al., Nature (2025)](https://doi.org/10.1038/s41586-024-08328-6). Checkpoint revisions are recorded in [the artifact manifest](artifacts/manifest.json).
