import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_task(y):
    unique_count = y.nunique(dropna=True)

    if y.dtype == "object" or str(y.dtype).startswith("category") or y.dtype == "bool":
        return "Classification"

    if pd.api.types.is_integer_dtype(y) and 2 <= unique_count <= 20:
        return "Classification"

    return "Regression"


def build_pipeline(task, model_choice, numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    if task == "Classification":
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(random_state=42)
    else:
        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def find_possible_leakage_columns(columns, target):
    leakage_terms = [
        "alive",
        "dead",
        "death",
        "died",
        "survive",
        "survived",
        "status",
        "outcome",
        "target",
        "label",
    ]
    target_text = str(target).lower()
    suspicious = []

    for column in columns:
        column_text = str(column).lower()
        if column_text == target_text:
            continue
        if any(term in column_text for term in leakage_terms):
            suspicious.append(column)

    return suspicious


def build_data_quality_report(df, target):
    rows = []
    feature_columns = [column for column in df.columns if column != target]

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        rows.append(
            {
                "Check": "Duplicate rows",
                "Column": "-",
                "Details": f"{duplicate_rows} duplicate rows found",
            }
        )

    for column in feature_columns:
        missing_rate = df[column].isnull().mean()
        unique_count = df[column].nunique(dropna=True)
        unique_rate = unique_count / len(df) if len(df) else 0

        if unique_count <= 1:
            rows.append(
                {
                    "Check": "Constant column",
                    "Column": column,
                    "Details": "Only one unique value",
                }
            )

        if missing_rate >= 0.4:
            rows.append(
                {
                    "Check": "High missing values",
                    "Column": column,
                    "Details": f"{missing_rate:.0%} missing",
                }
            )

        if unique_rate >= 0.95 and unique_count > 10:
            rows.append(
                {
                    "Check": "Possible ID column",
                    "Column": column,
                    "Details": f"{unique_count} unique values",
                }
            )

        if not pd.api.types.is_numeric_dtype(df[column]) and unique_count > 50 and unique_rate > 0.5:
            rows.append(
                {
                    "Check": "High-cardinality text",
                    "Column": column,
                    "Details": f"{unique_count} unique values",
                }
            )

    return pd.DataFrame(rows)


st.set_page_config(page_title="AutoML Lite", layout="wide")

st.title("AutoML Lite")
st.markdown("Upload your data, choose a target, train a model, and download predictions.")

st.sidebar.header("Settings")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
enable_cv = st.sidebar.checkbox("Run cross-validation", value=True)
enable_model_comparison = st.sidebar.checkbox("Compare models", value=True)

if file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Preview")
    st.dataframe(df.head(), width="stretch")

with col2:
    st.subheader("Model Settings")
    target = st.selectbox("Select target column", df.columns)

    inferred_task = infer_task(df[target])
    task = st.selectbox(
        "Task type",
        ["Classification", "Regression"],
        index=0 if inferred_task == "Classification" else 1,
    )

    if task == "Classification":
        model_choice = st.selectbox(
            "Choose a model",
            ["Logistic Regression", "Random Forest"],
        )
    else:
        model_choice = st.selectbox(
            "Choose a model",
            ["Linear Regression", "Random Forest"],
        )

    threshold = 0.5
    if task == "Classification" and df[target].nunique(dropna=True) == 2:
        threshold = st.slider(
            "Positive class threshold",
            0.05,
            0.95,
            0.5,
            0.05,
            help="Higher values make positive predictions stricter. Used only for binary classification.",
        )

    available_features = [column for column in df.columns if column != target]
    excluded_features = st.multiselect(
        "Exclude feature columns",
        available_features,
        help="Remove columns that should not be used for training, especially columns that leak the target.",
    )

st.subheader("Data Overview")
overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    st.metric("Rows", df.shape[0])

with overview_col2:
    st.metric("Columns", df.shape[1])

with overview_col3:
    st.metric("Missing values", int(df.isnull().sum().sum()))

missing_cols = df.isnull().sum()
missing_cols = missing_cols[missing_cols > 0]

if not missing_cols.empty:
    st.warning("Missing values detected. They will be imputed during training.")
    st.dataframe(missing_cols.rename("Missing values"), width="stretch")

quality_report = build_data_quality_report(df, target)
if not quality_report.empty:
    st.subheader("Data Quality Warnings")
    st.dataframe(quality_report, width="stretch")

X = df.drop(columns=[target] + excluded_features)
y = df[target]

if X.empty:
    st.error("The dataset must contain at least one feature column besides the target.")
    st.stop()

suspicious_columns = find_possible_leakage_columns(X.columns, target)
if suspicious_columns:
    st.warning(
        "Possible target leakage columns detected: "
        + ", ".join(str(column) for column in suspicious_columns)
        + ". Consider excluding them before training."
    )

if y.isnull().any():
    st.warning("Rows with missing target values will be removed before training.")
    valid_target_rows = y.notnull()
    X = X.loc[valid_target_rows]
    y = y.loc[valid_target_rows]

if len(X) < 2:
    st.error("The dataset must contain at least two valid rows for training.")
    st.stop()

if y.nunique(dropna=True) < 2:
    st.error("The target column must contain at least two unique values.")
    st.stop()

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

st.info(f"Detected task: {inferred_task}. Selected task: {task}.")

if task == "Classification":
    st.subheader("Target Distribution")
    target_counts = pd.Series(y).value_counts()
    st.bar_chart(target_counts)

    if len(target_counts) > 1 and target_counts.min() / target_counts.max() < 0.5:
        st.warning("Target is imbalanced. Accuracy may be misleading.")

if st.button("Train Model"):
    y_for_training = y
    target_encoder = None

    if task == "Classification":
        target_encoder = LabelEncoder()
        y_for_training = target_encoder.fit_transform(y.astype(str))
    else:
        y_for_training = pd.to_numeric(y, errors="coerce")
        if y_for_training.isnull().any():
            st.error(
                "Regression requires a numeric target column. "
                "Use Classification for text labels such as yes/no."
            )
            st.stop()

    stratify = None
    if task == "Classification":
        class_counts = pd.Series(y_for_training).value_counts()
        if class_counts.min() >= 2:
            stratify = y_for_training
        else:
            st.warning("Some classes have fewer than two rows, so the split cannot be stratified.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_for_training,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        st.error(f"Could not split the dataset: {exc}")
        st.stop()

    pipeline = build_pipeline(task, model_choice, numeric_features, categorical_features)

    split_col1, split_col2 = st.columns(2)
    split_col1.metric("Training rows", len(X_train))
    split_col2.metric("Testing rows", len(X_test))

    if task == "Classification":
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        distribution = pd.DataFrame(
            {
                "Train": train_counts,
                "Test": test_counts,
            }
        ).fillna(0).astype(int)
        distribution.index = target_encoder.inverse_transform(distribution.index.astype(int))
        st.subheader("Train/Test Class Distribution")
        st.dataframe(distribution, width="stretch")

    if enable_cv:
        try:
            if task == "Classification":
                min_class_count = pd.Series(y_for_training).value_counts().min()
                cv_splits = min(5, int(min_class_count))
                if cv_splits >= 2:
                    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(
                        pipeline,
                        X,
                        y_for_training,
                        cv=cv,
                        scoring="accuracy",
                    )
                    st.metric(
                        f"{cv_splits}-fold CV accuracy",
                        f"{cv_scores.mean():.2f}",
                        f"+/- {cv_scores.std():.2f}",
                    )
                else:
                    st.warning("Cross-validation skipped because at least one class has fewer than two rows.")
            else:
                cv_splits = min(5, len(X))
                if cv_splits >= 2:
                    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(
                        pipeline,
                        X,
                        y_for_training,
                        cv=cv,
                        scoring="neg_root_mean_squared_error",
                    )
                    rmse_scores = -cv_scores
                    st.metric(
                        f"{cv_splits}-fold CV RMSE",
                        f"{rmse_scores.mean():.2f}",
                        f"+/- {rmse_scores.std():.2f}",
                    )
        except Exception as exc:
            st.warning(f"Cross-validation skipped: {exc}")

    if enable_model_comparison:
        st.subheader("Model Comparison")
        comparison_rows = []
        comparison_choices = (
            ["Logistic Regression", "Random Forest"]
            if task == "Classification"
            else ["Linear Regression", "Random Forest"]
        )

        for comparison_model in comparison_choices:
            comparison_pipeline = build_pipeline(
                task,
                comparison_model,
                numeric_features,
                categorical_features,
            )
            try:
                comparison_pipeline.fit(X_train, y_train)
                comparison_preds = comparison_pipeline.predict(X_test)

                if task == "Classification":
                    score = accuracy_score(y_test, comparison_preds)
                    scoring_name = "Accuracy"
                else:
                    score = np.sqrt(mean_squared_error(y_test, comparison_preds))
                    scoring_name = "RMSE"

                comparison_rows.append(
                    {
                        "Model": comparison_model,
                        scoring_name: round(float(score), 4),
                    }
                )
            except Exception as exc:
                comparison_rows.append(
                    {
                        "Model": comparison_model,
                        "Error": str(exc),
                    }
                )

        st.dataframe(pd.DataFrame(comparison_rows), width="stretch")

    try:
        pipeline.fit(X_train, y_train)
        if task == "Classification" and len(target_encoder.classes_) == 2:
            positive_proba = pipeline.predict_proba(X_test)[:, 1]
            preds = (positive_proba >= threshold).astype(int)
            train_positive_proba = pipeline.predict_proba(X_train)[:, 1]
            train_preds = (train_positive_proba >= threshold).astype(int)
        else:
            preds = pipeline.predict(X_test)
            train_preds = pipeline.predict(X_train)
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.stop()

    st.subheader(f"Predictions for: {target}")
    st.success("Model trained successfully!")

    if task == "Classification":
        train_acc = accuracy_score(y_train, train_preds)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Accuracy", f"{acc:.2f}")
        metric_col2.metric("Precision", f"{precision:.2f}")
        metric_col3.metric("Recall", f"{recall:.2f}")
        metric_col4.metric("F1", f"{f1:.2f}")

        if train_acc - acc > 0.15:
            st.warning(
                f"Possible overfitting: train accuracy is {train_acc:.2f}, "
                f"but test accuracy is {acc:.2f}."
            )

        if len(target_encoder.classes_) == 2 and hasattr(pipeline, "predict_proba"):
            try:
                auc = roc_auc_score(y_test, positive_proba)
                st.metric("ROC AUC", f"{auc:.2f}")
            except Exception as exc:
                st.warning(f"ROC AUC could not be calculated: {exc}")

        labels_encoded = np.arange(len(target_encoder.classes_))
        labels = target_encoder.classes_
        matrix = confusion_matrix(y_test, preds, labels=labels_encoded)
        st.subheader("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(matrix, index=labels, columns=labels),
            width="stretch",
        )

        actual = target_encoder.inverse_transform(y_test)
        predicted = target_encoder.inverse_transform(preds)
    else:
        train_preds = pipeline.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("RMSE", f"{rmse:.2f}")
        metric_col2.metric("MAE", f"{mae:.2f}")
        metric_col3.metric("R2", f"{r2:.2f}")

        if rmse > train_rmse * 1.5:
            st.warning(
                f"Possible overfitting: train RMSE is {train_rmse:.2f}, "
                f"but test RMSE is {rmse:.2f}."
            )

        actual = y_test
        predicted = preds

    results = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predicted,
        }
    )

    if task == "Classification" and len(target_encoder.classes_) == 2:
        positive_class = target_encoder.classes_[1]
        results[f"Probability: {positive_class}"] = positive_proba

    st.subheader("Predictions")
    st.dataframe(results.head(50), width="stretch")

    trained_model = pipeline.named_steps["model"]
    if "Random Forest" in model_choice and hasattr(trained_model, "feature_importances_"):
        st.subheader("Feature Importance")
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.Series(trained_model.feature_importances_, index=feature_names)
        st.bar_chart(importance.sort_values(ascending=False).head(25))

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
