"""
House Price Prediction - Training Pipeline
Converted from the supplied Jupyter notebook.

Expected input:
    housing.csv

Target:
    median_house_value

Run:
    python train_model.py
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from preprocessing import CombinedAttributesAdder

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


DATA_FILE = Path("housing.csv")
MODEL_FILE = Path("final_model.pkl")
PIPELINE_FILE = Path("preprocessing_pipeline.pkl")
METADATA_FILE = Path("model_metadata.json")





def load_housing_data(path=DATA_FILE):
    return pd.read_csv(path)


def build_pipeline(training_features):
    num_attribs = list(training_features.select_dtypes(include=np.number).columns)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ])

    return full_pipeline, num_attribs


def stratified_split(housing):
    data = housing.copy()
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(data, data["income_cat"]))

    train_set = data.iloc[train_idx].drop(columns=["income_cat"])
    test_set = data.iloc[test_idx].drop(columns=["income_cat"])
    return train_set, test_set


def train_model(data_path=DATA_FILE):
    housing = load_housing_data(data_path)

    required = {
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "median_house_value", "ocean_proximity"
    }
    missing = required - set(housing.columns)
    if missing:
        raise ValueError(f"housing.csv is missing required columns: {sorted(missing)}")

    strat_train_set, strat_test_set = stratified_split(housing)

    X_train = strat_train_set.drop(columns=["median_house_value"])
    y_train = strat_train_set["median_house_value"].copy()

    pipeline, num_attribs = build_pipeline(X_train)
    X_train_prepared = pipeline.fit_transform(X_train)

    # Same Random Forest model family and GridSearchCV approach as the notebook.
    forest = RandomForestRegressor(random_state=42)
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    grid_search = GridSearchCV(
        forest,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train_prepared, y_train)

    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop(columns=["median_house_value"])
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = pipeline.transform(X_test)
    predictions = final_model.predict(X_test_prepared)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    # Store the feature ranges/categories needed by the frontend.
    numeric_ranges = {}
    for col in num_attribs:
        s = X_train[col]
        numeric_ranges[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        }

    metadata = {
        "target": "median_house_value",
        "best_params": grid_search.best_params_,
        "test_rmse": rmse,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numeric_ranges": numeric_ranges,
        "ocean_proximity_categories": sorted(
            X_train["ocean_proximity"].dropna().astype(str).unique().tolist()
        ),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test RMSE: ${rmse:,.2f}")
    print(f"Saved: {MODEL_FILE}, {PIPELINE_FILE}, {METADATA_FILE}")

    return final_model, pipeline, metadata


if __name__ == "__main__":
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "housing.csv was not found. Put the dataset beside train_model.py and run again."
        )
    train_model()
