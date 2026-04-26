# AutoML Lite

AutoML Lite is a Streamlit dashboard for quickly training and evaluating basic machine learning models from a CSV file. It is designed as a portfolio project that demonstrates practical data handling, model training, evaluation, and validation workflows using pandas and scikit-learn.

## Features

- Upload any CSV dataset through the browser.
- Select the target column and choose between classification or regression.
- Train Logistic Regression, Linear Regression, or Random Forest models.
- Automatic preprocessing with scikit-learn pipelines.
- Handles numeric and categorical features.
- Handles missing values through imputation.
- Adjustable train/test split.
- Cross-validation support.
- Model comparison table.
- Classification metrics: accuracy, precision, recall, F1, ROC AUC.
- Regression metrics: RMSE, MAE, R2.
- Confusion matrix for classification.
- Feature importance for Random Forest models.
- Prediction download as CSV.
- Data quality warnings for duplicate rows, constant columns, missing values, possible ID columns, and high-cardinality text.
- Possible target leakage warning.
- Binary classification threshold control and prediction probabilities.

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn

## Project Structure

```text
.
+-- automl.py
+-- requirements.txt
+-- README.md
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run The App

```powershell
streamlit run automl.py
```

If `streamlit` is not available as a command, use:

```powershell
python -m streamlit run automl.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How To Use

1. Upload a CSV file.
2. Select the target column.
3. Choose Classification or Regression.
4. Exclude any columns that should not be used for training.
5. Adjust the test size if needed.
6. Click Train Model.
7. Review metrics, class distribution, model comparison, feature importance, and predictions.
8. Download the prediction results as a CSV file.

## Example Use Case

For a target such as `alive?`, choose Classification. The app can train a model to predict whether a record belongs to the alive or not-alive class, then display classification metrics, confusion matrix, prediction probabilities, and possible overfitting warnings.

## Limitations

This project is intended as an educational and portfolio-level AutoML tool. It is not a replacement for a full production ML workflow.

Current limitations:

- Only supports a small set of baseline models.
- Does not perform advanced feature engineering.
- Does not tune hyperparameters.
- Does not persist trained models.
- Accuracy can be misleading if the dataset has leakage, duplicates, or severe class imbalance.
- Uploaded data is processed locally during the Streamlit session.

## Future Improvements

- Add hyperparameter tuning.
- Add model export with joblib.
- Add support for uploading new unseen data for prediction.
- Add ROC and precision-recall curve charts.
- Add SHAP or permutation importance.
- Add a sample dataset and screenshots.
