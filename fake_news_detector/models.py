from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def get_logistic_regression_model():
    """Returns a configured Logistic Regression model."""
    return LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, solver="liblinear", n_jobs=-1
    )


def get_xgboost_model():
    """Returns a configured XGBoost Classifier model."""
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )


def get_available_models():
    """Returns a dictionary of available models."""
    return {
        "XGBoost": get_xgboost_model(),
        "Logistic Regression": get_logistic_regression_model(),
    }


if __name__ == "__main__":
    lr_model = get_logistic_regression_model()
    xgb_model = get_xgboost_model()
    print("Logistic Regression model instance:", lr_model)
    print("XGBoost model instance:", xgb_model)
