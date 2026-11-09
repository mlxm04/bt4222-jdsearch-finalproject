# Personalised Re-Ranking of JD Search Results

BT4222 Group 10: Meredith Lim Xin Min, Jose Chow Yi, Li Tsz Wing Jemmia, Neo Zi Ning, Tham Brayden, Varidhi Goyal

We re-rank the products JD shows for each search (median 90 per list) using the user's past clicks, add-to-carts and orders. Data: [JDsearch](https://github.com/rucliujn/JDsearch), 173,831 searches made on 17 Oct.

## Results

| Model | Type | Test NDCG@10 | Notebook |
|---|---|---|---|
| Random | Baseline | 0.0804 | step 3 |
| MF | Baseline | 0.3168 | step 3 |
| NMF (Neural MF) | Baseline | 0.3267 | step 3 |
| NCF | Baseline | 0.3290 | step 3 |
| Popularity (90-day add-to-carts) | Baseline | 0.3329 | step 3 |
| KGIN, without features | Advanced | 0.3541 | step 4 |
| Features MLP (28 features only) | Advanced | 0.4021 | step 4 |
| LSTM + attention | Advanced | 0.4033 | step 4 |
| **KGIN + features** | Advanced | **0.4060** | step 4 |
| **LightGBM** | Advanced | **0.4063** | step 4 |

LightGBM and KGIN + features are tied (paired test, p = 0.78). Both beat popularity by 0.073 and the LSTM by 0.003 (p < 0.001).

## Notebooks

Run in this order. Each notebook reads the earlier steps' outputs from Google Drive and saves its own. All are saved with their outputs, so you can read the results without running anything.

| Step | Notebook | What it does | Colab runtime | Saves to |
|---|---|---|---|---|
| 1 | `step1_data_preprocess.ipynb` | Raw files → 5 tables; shuffles every list; 70/15/15 split | CPU | `processed/` |
| 2 | `step2_feature_engineering.ipynb` | 28 features (plus 8 same-day features, used only in one check) | CPU | `features/` |
| 3 | `step3_baselines.ipynb` | Random, popularity, MF, NCF, NMF, and why they fall short | T4 GPU | `baselines/` |
| 4 | `step4_lightgbm.ipynb` | LightGBM (LambdaRank) on the 28 features | CPU | `lightgbm/` |
| 4 | `step4_kgin.ipynb` | KGIN, with and without the 28 features | T4 GPU | `model/` |
| 4 | `step4_lstm_attention.ipynb` | LSTM over the user's history, with attention per shown product | T4 GPU | `advanced/` |
| 5 | `step5_compare_models.ipynb` | Paired tests and segments for the three step 4 models (no retraining) | CPU | `comparison/` |

The three step 4 notebooks can run in any order after steps 1–3. Step 5 runs last.

## Data

- **Raw:** `user_behavior_data.txt` and `product_meta_data.txt` from [JDsearch](https://github.com/rucliujn/JDsearch).
- **Processed data and all outputs:** [Google Drive folder](https://drive.google.com/drive/folders/1UrpMI0-ikTgQoQ4ZNwdPY5nvVD0j27CZ). Too large for GitHub (`history.parquet` alone is 421 MB).

## How to run

1. Put the two raw files in `MyDrive/BT4222_Group10_JDSearch/Raw Data/`.
2. Open a notebook in Colab (**File → Upload notebook**). For `step3_baselines`, `step4_kgin` and `step4_lstm_attention`, choose **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all** and allow Drive access.
4. Repeat for each notebook, in the order above.

`step4_kgin` takes longest: about 1¼ hours on a T4. `step4_lightgbm`'s saved outputs come from a local run of the same code; outside Colab it reads files from its own folder.

## Evaluation (same in every notebook)

- **Searches:** the 171,728 with at least one interaction (2,103 without are left out).
- **Split:** by search, 70/15/15, seed 4222: 120,209 train, 25,759 validation, 25,760 test. Each user has one search, so no user is in two splits.
- **Lists:** every shown product. The raw data stores interacted products first, so every list is shuffled and ties are broken at random.
- **Metric:** NDCG@10, with click = 1, add-to-cart = 3, order = 7.
- **Choices:** settings are picked on validation; test is scored only after that.
- **Timing:** searches have no time of day, so models only use other users' activity from before 17 Oct. The 8 same-day features break this rule and are used only in one check.

## Google Drive layout

```
BT4222_Group10_JDSearch/
├── Raw Data/    the two raw JDsearch files
├── processed/   step 1 tables (searches, exposures, history, products, titles)
├── features/    step 2 features, one file per group
├── baselines/   step 3 scores, results, MF / NCF / NMF weights
├── lightgbm/    step 4 LightGBM: models, scores, results
├── model/       step 4 KGIN: weights, scores, results, ablations
├── advanced/    step 4 LSTM: weights, scores, results, ablations
├── comparison/  step 5 results, paired tests, segments
└── step*.ipynb  copies of the notebooks
```

Every score file has one row per shown product (`sid`, `wid`, `score`) in `processed/exposures.parquet` order, so any two models can be compared row by row.
