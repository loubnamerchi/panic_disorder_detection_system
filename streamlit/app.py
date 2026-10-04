
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)

CATEGORIES_PATH = Path("categorical_values.yaml")

CATEGORICAL_COLUMNS = [
    "Gender",
    "Family History",
    "Personal History",
    "Current Stressors",
    "Symptoms",
    "Severity",
    "Impact on Life",
    "Demographics",
    "Medical History",
    "Psychiatric History",
    "Substance Use",
    "Coping Mechanisms",
    "Social Support",
    "Lifestyle Factors",
]

REQUIRED_COLUMNS = [
    "Age",
    *CATEGORICAL_COLUMNS,
]


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Panic Disorder Detection",
    page_icon="🧠",
    layout="wide",
)


# =============================================================================
# LOAD CATEGORICAL VALUES
# =============================================================================

@st.cache_data
def load_category_options() -> dict[str, list[str]]:
    """Load categorical options from categorical_values.yaml."""

    if not CATEGORIES_PATH.exists():
        raise FileNotFoundError(
            f"Categorical values file not found: {CATEGORIES_PATH}"
        )

    with CATEGORIES_PATH.open("r", encoding="utf-8") as file:
        categories = yaml.safe_load(file)

    if not isinstance(categories, dict):
        raise ValueError(
            "categorical_values.yaml must contain a mapping of "
            "column names to lists of values."
        )

    for column in CATEGORICAL_COLUMNS:
        if column not in categories:
            raise ValueError(
                f"Missing categorical column '{column}' "
                "in categorical_values.yaml."
            )

        if not isinstance(categories[column], list) or not categories[column]:
            raise ValueError(
                f"'{column}' must contain at least one categorical value."
            )

    return {
        column: [str(value) for value in categories[column]]
        for column in CATEGORICAL_COLUMNS
    }


try:
    CATEGORY_OPTIONS = load_category_options()
    CATEGORY_LOAD_ERROR = None
except Exception as exc:
    CATEGORY_OPTIONS = {}
    CATEGORY_LOAD_ERROR = str(exc)


# =============================================================================
# API HELPERS
# =============================================================================

def get_api_info() -> dict:
    response = requests.get(
        f"{API_URL}/info",
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def get_api_health() -> dict:
    response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("🧠 Panic Disorder Detection")
    st.caption(f"API: `{API_URL}`")

    # -------------------------------------------------------------------------
    # API / MODEL INFORMATION
    # -------------------------------------------------------------------------

    try:
        info = get_api_info()

        st.success("API connected")

        st.metric(
            "Features",
            info["feature_count"],
        )

        with st.expander("Model metrics"):

            metrics = info.get("metrics", {})

            col1, col2 = st.columns(2)

            col1.metric(
                "ROC-AUC",
                f"{metrics.get('roc_auc', 0):.4f}",
            )

            col1.metric(
                "PR-AUC",
                f"{metrics.get('pr_auc', 0):.4f}",
            )

            col1.metric(
                "Precision",
                f"{metrics.get('precision', 0):.4f}",
            )

            col2.metric(
                "Recall",
                f"{metrics.get('recall', 0):.4f}",
            )

            col2.metric(
                "F1",
                f"{metrics.get('f1', 0):.4f}",
            )

            col2.metric(
                "Brier Score",
                f"{metrics.get('brier_score', 0):.4f}",
            )

    except requests.exceptions.RequestException:
        st.error("API unreachable")

    st.divider()

    mode = st.radio(
        "Mode",
        [
            "Single prediction",
            "Batch prediction",
            "API Health",
        ],
    )


# =============================================================================
# CATEGORY ERROR
# =============================================================================

if CATEGORY_LOAD_ERROR:

    st.error(
        "Unable to load categorical values from "
        "`/categorical_values.yaml`."
    )

    st.code(CATEGORY_LOAD_ERROR)

    st.stop()


# =============================================================================
# SINGLE PREDICTION
# =============================================================================

if mode == "Single prediction":

    st.header("Single Panic Disorder Prediction")

    st.write(
        "Enter the patient's demographic, psychological, behavioral, "
        "and clinical characteristics."
    )

    with st.form("prediction_form"):

        # ---------------------------------------------------------------------
        # DEMOGRAPHIC INFORMATION
        # ---------------------------------------------------------------------

        st.subheader("Demographic Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=34,
                step=1,
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                CATEGORY_OPTIONS["Gender"],
            )

        # ---------------------------------------------------------------------
        # MEDICAL / PSYCHOLOGICAL HISTORY
        # ---------------------------------------------------------------------

        st.subheader("Medical and Psychological History")

        col1, col2 = st.columns(2)

        with col1:

            family_history = st.selectbox(
                "Family History",
                CATEGORY_OPTIONS["Family History"],
            )

            personal_history = st.selectbox(
                "Personal History",
                CATEGORY_OPTIONS["Personal History"],
            )

            medical_history = st.selectbox(
                "Medical History",
                CATEGORY_OPTIONS["Medical History"],
            )

            psychiatric_history = st.selectbox(
                "Psychiatric History",
                CATEGORY_OPTIONS["Psychiatric History"],
            )

        with col2:

            substance_use = st.selectbox(
                "Substance Use",
                CATEGORY_OPTIONS["Substance Use"],
            )

            social_support = st.selectbox(
                "Social Support",
                CATEGORY_OPTIONS["Social Support"],
            )

            coping_mechanisms = st.selectbox(
                "Coping Mechanisms",
                CATEGORY_OPTIONS["Coping Mechanisms"],
            )

        # ---------------------------------------------------------------------
        # SYMPTOMS / STRESS / SEVERITY
        # ---------------------------------------------------------------------

        st.subheader("Symptoms and Psychological Factors")

        col1, col2 = st.columns(2)

        with col1:

            current_stressors = st.selectbox(
                "Current Stressors",
                CATEGORY_OPTIONS["Current Stressors"],
            )

            symptoms = st.selectbox(
                "Symptoms",
                CATEGORY_OPTIONS["Symptoms"],
            )

            severity = st.selectbox(
                "Severity",
                CATEGORY_OPTIONS["Severity"],
            )

        with col2:

            impact_on_life = st.selectbox(
                "Impact on Life",
                CATEGORY_OPTIONS["Impact on Life"],
            )

            demographics = st.selectbox(
                "Demographics",
                CATEGORY_OPTIONS["Demographics"],
            )

            lifestyle_factors = st.selectbox(
                "Lifestyle Factors",
                CATEGORY_OPTIONS["Lifestyle Factors"],
            )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
        )

    # =========================================================================
    # SEND REQUEST TO FASTAPI
    # =========================================================================

    if submitted:

        payload = {
            "age": age,
            "gender": gender,
            "family_history": family_history,
            "personal_history": personal_history,
            "current_stressors": current_stressors,
            "symptoms": symptoms,
            "severity": severity,
            "impact_on_life": impact_on_life,
            "demographics": demographics,
            "medical_history": medical_history,
            "psychiatric_history": psychiatric_history,
            "substance_use": substance_use,
            "coping_mechanisms": coping_mechanisms,
            "social_support": social_support,
            "lifestyle_factors": lifestyle_factors,
        }

        with st.spinner("Running prediction..."):

            try:

                start = time.perf_counter()

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=10,
                )

                latency = (
                    time.perf_counter() - start
                ) * 1000

                response.raise_for_status()

                result = response.json()

                # -------------------------------------------------------------
                # RESULTS
                # -------------------------------------------------------------

                st.divider()

                col1, col2, col3 = st.columns(3)

                if result["is_panic_disorder"]:
                    col1.error("PANIC DISORDER PREDICTED")
                else:
                    col1.success("NO PANIC DISORDER PREDICTED")

                col2.metric(
                    "Probability",
                    f"{result['probability']:.2%}",
                )

                col3.metric(
                    "Latency",
                    f"{latency:.0f} ms",
                )

                # -------------------------------------------------------------
                # PROBABILITY
                # -------------------------------------------------------------

                st.subheader("Prediction Probability")

                probability = float(result["probability"])

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                # -------------------------------------------------------------
                # RAW RESPONSE
                # -------------------------------------------------------------

                with st.expander("Raw API response"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the FastAPI server."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "The API request timed out."
                )

            except requests.exceptions.HTTPError as exc:

                st.error(
                    f"API error: {exc}"
                )

                try:
                    st.json(response.json())
                except Exception:
                    pass

            except Exception as exc:
                st.error(
                    f"Error: {exc}"
                )


# =============================================================================
# BATCH PREDICTION
# =============================================================================

elif mode == "Batch prediction":

    st.header("Batch Panic Disorder Prediction")

    st.write(
        "Upload a CSV file containing patient records."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded:

        try:

            df = pd.read_csv(uploaded)

            st.subheader("Uploaded Data")

            st.dataframe(
                df.head(10),
                use_container_width=True,
            )

            # -----------------------------------------------------------------
            # VALIDATE COLUMNS
            # -----------------------------------------------------------------

            missing = [
                column
                for column in REQUIRED_COLUMNS
                if column not in df.columns
            ]

            if missing:

                st.error(
                    f"Missing columns: {missing}"
                )

            elif len(df) == 0:

                st.warning("The uploaded CSV is empty.")

            elif st.button(
                "Predict all patients",
                use_container_width=True,
            ):

                all_predictions = []
                all_probabilities = []

                progress = st.progress(0)

                batch_size = 1000

                n_batches = (
                    len(df) + batch_size - 1
                ) // batch_size

                with st.spinner(
                    f"Predicting {len(df):,} patients..."
                ):

                    start = time.perf_counter()

                    for i in range(n_batches):

                        start_row = i * batch_size
                        end_row = start_row + batch_size

                        chunk = df.iloc[start_row:end_row]

                        payload = {
                            "requests": chunk[
                                REQUIRED_COLUMNS
                            ].rename(
                                columns={"Age": "age"}
                            ).rename(
                                columns={
                                    column: column.lower().replace(
                                        " ",
                                        "_",
                                    )
                                    for column in CATEGORICAL_COLUMNS
                                }
                            ).to_dict(
                                orient="records"
                            )
                        }

                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            json=payload,
                            timeout=60,
                        )

                        response.raise_for_status()

                        result = response.json()

                        all_predictions.extend(
                            result["predictions"]
                        )

                        all_probabilities.extend(
                            result["probabilities"]
                        )

                        progress.progress(
                            (i + 1) / n_batches
                        )

                    latency = (
                        time.perf_counter() - start
                    ) * 1000

                # -------------------------------------------------------------
                # ADD RESULTS
                # -------------------------------------------------------------

                df["prediction"] = all_predictions

                df["probability"] = all_probabilities

                df["is_panic_disorder"] = (
                    df["prediction"] == 1
                )

                panic_count = int(
                    df["is_panic_disorder"].sum()
                )

                # -------------------------------------------------------------
                # SUMMARY
                # -------------------------------------------------------------

                st.divider()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Patients",
                    f"{len(df):,}",
                )

                col2.metric(
                    "Panic Disorder",
                    f"{panic_count:,}",
                )

                col3.metric(
                    "Prediction Rate",
                    f"{panic_count / len(df):.2%}",
                )

                col4.metric(
                    "Latency",
                    f"{latency:.0f} ms",
                )

                # -------------------------------------------------------------
                # RESULTS TABLE
                # -------------------------------------------------------------

                st.subheader("Prediction Results")

                st.dataframe(
                    df[
                        [
                            "Age",
                            "prediction",
                            "probability",
                            "is_panic_disorder",
                        ]
                    ],
                    use_container_width=True,
                )

                # -------------------------------------------------------------
                # DOWNLOAD
                # -------------------------------------------------------------

                st.download_button(
                    "Download predictions CSV",
                    data=df.to_csv(index=False),
                    file_name="panic_disorder_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI server."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The API request timed out."
            )

        except requests.exceptions.HTTPError as exc:
            st.error(
                f"API error: {exc}"
            )

        except Exception as exc:
            st.error(
                f"Error processing file: {exc}"
            )


# =============================================================================
# API HEALTH
# =============================================================================

elif mode == "API Health":

    st.header("API Health")

    if st.button("Refresh"):
        st.rerun()

    try:

        health = get_api_health()

        if health["status"] == "ok":
            st.success(
                f"Status: {health['status'].upper()}"
            )
        else:
            st.warning(
                f"Status: {health['status'].upper()}"
            )

        st.metric(
            "Model Loaded",
            "Yes" if health["model_loaded"] else "No",
        )

        # ---------------------------------------------------------------------
        # MODEL INFORMATION
        # ---------------------------------------------------------------------

        st.subheader("Model Information")

        info = get_api_info()

        st.write(
            f"**Feature count:** {info['feature_count']}"
        )

        with st.expander("Features used by the model"):
            st.write(info["features"])

        with st.expander("Evaluation metrics"):
            st.json(info["metrics"])

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the FastAPI server."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out."
        )

    except requests.exceptions.HTTPError as exc:
        st.error(
            f"API error: {exc}"
        )

    except Exception as exc:
        st.error(
            f"API error: {exc}"
        )

