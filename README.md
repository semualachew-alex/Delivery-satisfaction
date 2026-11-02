# Maternal Delivery-Care Satisfaction Predictor

A machine-learning course project exploring maternal satisfaction with vaginal and cesarean
delivery-care services in health facilities in Bahir Dar, Northwest Ethiopia. It includes an
annotated analysis notebook and a Streamlit app that demonstrates predictions from a logistic
regression model.

> **Educational use only.** This project is not a clinically validated prediction tool and must
> not be used to make decisions about patient care, facility performance, or individuals.

## Project overview

The notebook covers:

- Descriptive statistics and visualizations
- Data cleaning, one-hot encoding, and standardization
- An 80/20 train/test split
- ANOVA F-test feature selection with `SelectKBest`
- Logistic regression training and evaluation
- Coefficient and permutation feature-importance analysis

The trained model and the preprocessing components needed for inference are stored in the output
of the notebook's final cell. The Streamlit app reads that embedded bundle directly, so separate
`.joblib` files are not needed.

## Dataset and source

The included SPSS dataset, `Delivery care satisfaction.sav`, contains responses from 894 mothers
(448 vaginal and 446 cesarean deliveries) interviewed in 2018. The outcome is `satisfaction`
(1 = satisfied, 0 = not satisfied); 61.4% of respondents were satisfied.

Source study: Karoni HF, Bantie GM, Azage M, Kasa AS, Aynie AA, Tsegaye GW. “Maternal satisfaction
among vaginal and cesarean section delivery care services in Bahir Dar city health facilities,
Northwest Ethiopia: a facility-based comparative cross-sectional study.” *BMC Pregnancy and
Childbirth*. 2020;20:473. [https://doi.org/10.1186/s12884-020-03170-w](https://doi.org/10.1186/s12884-020-03170-w)

Dataset credit: Getasew Mulat and co-authors.

## Test-set results

Results reported by the notebook on the 20% hold-out test set (179 records):

| Metric | Score |
|---|---:|
| Accuracy | 0.978 |
| Precision | 0.983 |
| Recall | 0.983 |
| F1 score | 0.983 |
| ROC AUC | 0.994 |
| Brier score | 0.024 |

These scores should be interpreted with care: selected service ratings are also components of the
original satisfaction score. The high performance therefore largely reflects agreement with the
score's own components, rather than evidence of generalizable or causal prediction.

## Repository contents

| File | Description |
|---|---|
| `The_last_maternal_satisfaction_pridiction_model.ipynb` | Analysis, model training, evaluation, and embedded model bundle |
| `Delivery care satisfaction.sav` | Input survey dataset in SPSS format |
| `app8.py` | Streamlit prediction interface |
| `requirements.txt` | Python package dependencies |

Keep the notebook and app in the same directory so the app can load the embedded model.

## Run the app locally

You will need Python and pip. From the project directory, create and activate a virtual environment
(recommended), then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the requirements and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app8.py
```

Open the local URL printed by Streamlit (usually <http://localhost:8501>). Answer the ten service
questions and select **Predict Satisfaction** to see the estimated probability and the factors
contributing to the result.

## Run or retrain the notebook

1. Install the project requirements in your Python environment.
2. Open `The_last_maternal_satisfaction_pridiction_model.ipynb` in Jupyter or VS Code.
3. Keep `Delivery care satisfaction.sav` in the project directory; the notebook reads it by that
   filename.
4. Run the notebook cells in order. The final cell embeds the trained model and preprocessing
   bundle in the notebook's output.
5. Save the notebook after the final cell has run. The app uses that saved output.

If you retrain the model, update and save the notebook before running the app so it uses the newly
embedded model.

## Deploy with Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py` as the
   app's main file.
3. Deploy. Ensure the notebook, dataset, and `requirements.txt` are included at the repository root.

## Responsible use and data handling

This project is intended for learning and demonstration. The model has not been validated for
clinical, operational, or individual-level decisions. Review the source study and applicable data
permissions before redistributing or reusing the dataset. Only load notebook files and embedded
model outputs from sources you trust.

## Author

Alachew Semu, Data Science, and 2026.
