from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
import shap
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/model_bundle.joblib")


def train() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=500,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=4,
            min_child_samples=30,
            subsample=1.0,
            colsample_bytree=0.7,
            reg_alpha=5.0,
            reg_lambda=5.0,
            random_state=42,
            verbose=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on hold-out split
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)
    print(f"Hold-Out ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    print(f"Hold-Out Recall:  {recall_score(y_test, y_preds):.4f}")

    # Build SHAP TreeExplainer & extract clean feature names
    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    fitted_lgbm = pipeline.named_steps["classifier"]
    raw_names = fitted_preprocessor.named_steps["transformer"].get_feature_names_out()
    clean_feature_names = [name.split("__")[-1] for name in raw_names]
    explainer = shap.TreeExplainer(fitted_lgbm)

    # Save everything together in one bundle
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "explainer": explainer,
        "feature_names": clean_feature_names,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle successfully to {MODEL_PATH}")


if __name__ == "__main__":
    train()
