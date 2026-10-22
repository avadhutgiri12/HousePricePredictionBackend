"""
House Price Prediction System - Streamlit Frontend

Run:
    pip install -r requirements.txt
    streamlit run app.py

The app can:
1. Train the model from housing.csv if model files do not exist.
2. Accept property inputs.
3. Return an estimated house value.
4. Estimate model confidence from Random Forest tree agreement.
5. Show a prediction interval based on the tree prediction distribution.
6. Show model/training information.
"""

from pathlib import Path
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "final_model.pkl"
PIPELINE_FILE = BASE / "preprocessing_pipeline.pkl"
METADATA_FILE = BASE / "model_metadata.json"
DATA_FILE = BASE / "housing.csv"


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: .35rem;
        font-size: 2.5rem;
    }

    .hero p {
        color: #cbd5e1;
        margin: 0;
    }

    .result {
        padding: 1.5rem;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin: 1rem 0;
    }

    .result-label {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #000000;
    }

    .confidence-label {
        color: #475569;
        font-size: 1rem;
        margin-top: 18px;
    }

    .confidence-value {
        color: #000000;
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 4px;
    }

    .range-box {
        padding: 1rem;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Prediction</h1>
        <p>
            Machine-learning powered property valuation using the
            California housing prediction pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    if not (MODEL_FILE.exists() and PIPELINE_FILE.exists()):
        return None, None, None

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    metadata = (
        json.loads(METADATA_FILE.read_text())
        if METADATA_FILE.exists()
        else {}
    )

    return model, pipeline, metadata


def train_if_needed():
    if MODEL_FILE.exists() and PIPELINE_FILE.exists():
        return True

    if not DATA_FILE.exists():
        return False

    with st.spinner("Training the model for the first time..."):
        result = subprocess.run(
            [sys.executable, str(BASE / "train_model.py")],
            cwd=BASE,
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        st.error("Model training failed.")
        st.code(result.stderr or result.stdout)
        return False

    st.cache_resource.clear()
    return True


if not train_if_needed():
    st.warning(
        "No trained model was found. Put `housing.csv` in this project "
        "folder, then refresh the page to train the model."
    )
    st.stop()


model, pipeline, metadata = load_artifacts()

if model is None or pipeline is None:
    st.error(
        "The model could not be loaded. Delete the existing `.pkl` files "
        "and retrain the model using `python train_model.py`."
    )
    st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Model")

    st.success("RandomForestRegressor Model ready")

    if metadata:
        st.metric(
            "Training rows",
            f"{metadata.get('train_rows', 0):,}",
        )

        st.metric(
            "Test RMSE",
            f"${metadata.get('test_rmse', 0):,.0f}",
        )

        st.caption(
            "Random Forest Regressor with GridSearchCV"
        )

    st.divider()

    st.caption(
        "Input ranges are based on the supplied training dataset."
    )


# ---------- Input section ----------
st.subheader("Property Details")

ranges = metadata.get("numeric_ranges", {}) if metadata else {}

categories = metadata.get(
    "ocean_proximity_categories",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN",
    ],
)

c1, c2, c3 = st.columns(3)


with c1:
    longitude = st.number_input(
        "Longitude",
        min_value=float(
            ranges.get("longitude", {}).get(
                "min", -124.35
            )
        ),
        max_value=float(
            ranges.get("longitude", {}).get(
                "max", -114.31
            )
        ),
        value=float(
            ranges.get("longitude", {}).get(
                "median", -118.49
            )
        ),
        format="%.4f",
    )

    latitude = st.number_input(
        "Latitude",
        min_value=float(
            ranges.get("latitude", {}).get(
                "min", 32.54
            )
        ),
        max_value=float(
            ranges.get("latitude", {}).get(
                "max", 41.95
            )
        ),
        value=float(
            ranges.get("latitude", {}).get(
                "median", 34.26
            )
        ),
        format="%.4f",
    )

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=float(
            ranges.get("housing_median_age", {}).get(
                "min", 1
            )
        ),
        max_value=float(
            ranges.get("housing_median_age", {}).get(
                "max", 52
            )
        ),
        value=float(
            ranges.get("housing_median_age", {}).get(
                "median", 29
            )
        ),
        step=1.0,
    )


with c2:
    total_rooms = st.number_input(
        "Total Rooms",
        min_value=float(
            ranges.get("total_rooms", {}).get(
                "min", 2
            )
        ),
        max_value=float(
            ranges.get("total_rooms", {}).get(
                "max", 39320
            )
        ),
        value=float(
            ranges.get("total_rooms", {}).get(
                "median", 2127
            )
        ),
        step=1.0,
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=float(
            ranges.get("total_bedrooms", {}).get(
                "min", 1
            )
        ),
        max_value=float(
            ranges.get("total_bedrooms", {}).get(
                "max", 6445
            )
        ),
        value=float(
            ranges.get("total_bedrooms", {}).get(
                "median", 435
            )
        ),
        step=1.0,
    )

    population = st.number_input(
        "Population",
        min_value=float(
            ranges.get("population", {}).get(
                "min", 3
            )
        ),
        max_value=float(
            ranges.get("population", {}).get(
                "max", 35682
            )
        ),
        value=float(
            ranges.get("population", {}).get(
                "median", 1166
            )
        ),
        step=1.0,
    )


with c3:
    households = st.number_input(
        "Households",
        min_value=float(
            ranges.get("households", {}).get(
                "min", 1
            )
        ),
        max_value=float(
            ranges.get("households", {}).get(
                "max", 6082
            )
        ),
        value=float(
            ranges.get("households", {}).get(
                "median", 409
            )
        ),
        step=1.0,
    )

    median_income = st.number_input(
        "Median Income",
        min_value=float(
            ranges.get("median_income", {}).get(
                "min", 0.4999
            )
        ),
        max_value=float(
            ranges.get("median_income", {}).get(
                "max", 15.0001
            )
        ),
        value=float(
            ranges.get("median_income", {}).get(
                "median", 3.5348
            )
        ),
        step=0.0001,
        format="%.4f",
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        categories,
    )


st.divider()


# ---------- Prediction ----------
if st.button(
    "Predict House Price",
    type="primary",
    use_container_width=True,
):
    input_df = pd.DataFrame(
        [
            {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }
        ]
    )

    try:
        # Transform user input using the exact preprocessing pipeline
        prepared = pipeline.transform(input_df)

        # Main Random Forest prediction
        prediction = float(model.predict(prepared)[0])

        # -------------------------------------------------
        # Confidence / uncertainty estimation
        # -------------------------------------------------
        #
        # RandomForestRegressor does not provide a true
        # probability/confidence score for regression.
        #
        # Instead, we calculate the prediction from every
        # decision tree and measure how much those trees
        # disagree.
        #
        # Lower disagreement = higher model confidence.
        # -------------------------------------------------

        if hasattr(model, "estimators_") and len(model.estimators_) > 0:

            tree_predictions = np.array(
                [
                    float(tree.predict(prepared)[0])
                    for tree in model.estimators_
                ]
            )

            # Mean prediction across all trees
            prediction = float(
                np.mean(tree_predictions)
            )

            # Standard deviation of tree predictions
            prediction_std = float(
                np.std(tree_predictions)
            )

            # Approximate 95% prediction interval
            lower_bound = max(
                0.0,
                prediction - 1.96 * prediction_std,
            )

            upper_bound = (
                prediction + 1.96 * prediction_std
            )

            # Relative uncertainty
            relative_uncertainty = (
                prediction_std
                / max(abs(prediction), 1.0)
            )

            # Convert uncertainty to an easy-to-understand
            # confidence indicator.
            confidence = 100.0 * (
                1.0 - min(relative_uncertainty, 0.95)
            )

            confidence = max(
                0.0,
                min(confidence, 100.0),
            )

        else:
            prediction_std = 0.0
            lower_bound = prediction
            upper_bound = prediction
            confidence = 0.0


        # -------------------------------------------------
        # Prediction analysis
        # -------------------------------------------------

        st.subheader("Prediction Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Price",
                f"${prediction:,.0f}",
            )

        with col2:
            st.metric(
                "Lower Estimate",
                f"${lower_bound:,.0f}",
            )

        with col3:
            st.metric(
                "Upper Estimate",
                f"${upper_bound:,.0f}",
            )

        # -------------------------------------------------
        # Confidence progress bar
        # -------------------------------------------------

        st.write("Model Confidence")

        st.progress(
            int(round(confidence)),
            text=f"{confidence:.1f}% confidence",
        )

        # -------------------------------------------------
        # Prediction interval
        # -------------------------------------------------

        

        

        st.caption(
            "Important: this confidence value is an uncertainty "
            "indicator derived from Random Forest tree agreement, "
            "not a statistically calibrated probability. The "
            "prediction range is also an approximate model-based "
            "interval and should not be interpreted as a guaranteed "
            "market-value range."
        )

    except Exception as exc:
        st.error(
            f"Prediction failed: {exc}"
        )


# ---------- Model details ----------
with st.expander("Model details"):
    if metadata:
        st.write(
            "Best hyperparameters:",
            metadata.get("best_params"),
        )

        st.write(
            "Test RMSE:",
            f"${metadata.get('test_rmse', 0):,.2f}",
        )

        st.write(
            "Training rows:",
            metadata.get("train_rows"),
        )

        st.write(
            "Test rows:",
            metadata.get("test_rows"),
        )
