import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yaml
import sys

# =============================================================================
# SETTINGS
# =============================================================================

API_URL = os.getenv("API_URL", "https://panic-disorder-detection-system.onrender.com")
#API_URL = "https://panic-disorder-detection-system.onrender.com"
CATEGORIES_FILE = Path("categorical_values.yaml")

BATCH_SIZE = 1000



# =============================================================================
# PAGE
# =============================================================================

st.set_page_config(
    page_title="Panic Disorder Detection",
    page_icon="🧠",
    layout="wide",
)


# =============================================================================
# LOAD DROPDOWN VALUES
# =============================================================================
# Streamlit only needs these values to build the form.
# FastAPI still validates and processes the submitted values.

@st.cache_data
def load_categories():
    with open(CATEGORIES_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


try:
    categories = load_categories()
except Exception as e:
    st.error(f"Could not load categorical_values.yaml: {e}")
    st.stop()


# =============================================================================
# API HELPERS
# =============================================================================

def get_api_info():
    response = requests.get(
        f"{API_URL}/info",
        timeout=160,
    )

    response.raise_for_status()
    return response.json()


def get_api_health():
    response = requests.get(
        f"{API_URL}/health",
        timeout=160,
    )

    response.raise_for_status()
    return response.json()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("🧠 Panic Disorder Detection")
    st.caption(f"API: `{API_URL}`")

    try:
        info = get_api_info()

        st.success("API connected")

        st.metric(
            "Features",
            info["feature_count"],
        )

        with st.expander("Model metrics"):

            metrics = info["metrics"]

            col1, col2 = st.columns(2)

            col1.metric(
                "ROC-AUC",
                f"{metrics['roc_auc']:.4f}",
            )

            col1.metric(
                "PR-AUC",
                f"{metrics['pr_auc']:.4f}",
            )

            col1.metric(
                "Precision",
                f"{metrics['precision']:.4f}",
            )

            col2.metric(
                "Recall",
                f"{metrics['recall']:.4f}",
            )

            col2.metric(
                "F1",
                f"{metrics['f1']:.4f}",
            )

            col2.metric(
                "Brier Score",
                f"{metrics['brier_score']:.4f}",
            )

    except requests.exceptions.ConnectionError:
        st.error("API unreachable")
        

    
    except requests.exceptions.Timeout:
        st.error("API request timed out")
    

    

    except Exception as e:
        st.error(f"API error: {e}")

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
# SINGLE PREDICTION
# =============================================================================

if mode == "Single prediction":

    st.header("Single Panic Disorder Prediction")

    st.write(
        "Enter the participant information and send it to the FastAPI model."
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
                categories["Gender"],
            )

        # ---------------------------------------------------------------------
        # HISTORY
        # ---------------------------------------------------------------------

        st.subheader("Medical and Psychological History")

        col1, col2 = st.columns(2)

        with col1:

            family_history = st.selectbox(
                "Family History",
                categories["Family History"],
            )

            personal_history = st.selectbox(
                "Personal History",
                categories["Personal History"],
            )

            medical_history = st.selectbox(
                "Medical History",
                categories["Medical History"],
            )

            psychiatric_history = st.selectbox(
                "Psychiatric History",
                categories["Psychiatric History"],
            )

        with col2:

            substance_use = st.selectbox(
                "Substance Use",
                categories["Substance Use"],
            )

            social_support = st.selectbox(
                "Social Support",
                categories["Social Support"],
            )

            coping_mechanisms = st.selectbox(
                "Coping Mechanisms",
                categories["Coping Mechanisms"],
            )

        # ---------------------------------------------------------------------
        # SYMPTOMS AND LIFESTYLE
        # ---------------------------------------------------------------------

        st.subheader("Symptoms and Lifestyle")

        col1, col2 = st.columns(2)

        with col1:

            current_stressors = st.selectbox(
                "Current Stressors",
                categories["Current Stressors"],
            )

            symptoms = st.selectbox(
                "Symptoms",
                categories["Symptoms"],
            )

            severity = st.selectbox(
                "Severity",
                categories["Severity"],
            )

        with col2:

            impact_on_life = st.selectbox(
                "Impact on Life",
                categories["Impact on Life"],
            )

            demographics = st.selectbox(
                "Demographics",
                categories["Demographics"],
            )

            lifestyle_factors = st.selectbox(
                "Lifestyle Factors",
                categories["Lifestyle Factors"],
            )

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True,
        )


    # -------------------------------------------------------------------------
    # CALL FASTAPI
    # -------------------------------------------------------------------------

    if submitted:

        # These names match PanicDisorderRequest in schemas.py.
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

        try:

            with st.spinner("Running prediction..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=160,
                )

            response.raise_for_status()

            result = response.json()

            # -----------------------------------------------------------------
            # RESULTS
            # -----------------------------------------------------------------

            st.divider()

            col1, col2 = st.columns(2)

            if result["is_panic_disorder"]:
                col1.error("PANIC DISORDER PREDICTED")
            else:
                col1.success("NO PANIC DISORDER PREDICTED")

            col2.metric(
                "Probability",
                f"{result['probability']:.2%}",
            )

            # -----------------------------------------------------------------
            # PROBABILITY
            # -----------------------------------------------------------------

            st.subheader("Prediction Probability")

            st.progress(
                result["probability"]
            )

            # -----------------------------------------------------------------
            # API RESPONSE
            # -----------------------------------------------------------------

            with st.expander("API response"):

                st.json(result)

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to the FastAPI server."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The API request timed out."
            )

        except requests.exceptions.HTTPError:

            st.error(
                f"API returned an error: {response.status_code}"
            )

            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)

        except Exception as e:

            st.error(f"Error: {e}")


# =============================================================================
# BATCH PREDICTION
# =============================================================================

elif mode == "Batch prediction":

    st.header("Batch Panic Disorder Prediction")

    st.write(
        "Upload a CSV file and Streamlit will send the records to FastAPI."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.dataframe(
                df.head(10),
                use_container_width=True,
            )

            # These are the columns expected by the FastAPI request.
            required_columns = [
                "Age",
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

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing columns: {missing_columns}"
                )

            elif len(df) == 0:

                st.warning("The uploaded CSV is empty.")

            else:

                if st.button(
                    "Predict all patients",
                    use_container_width=True,
                ):

                    predictions = []
                    probabilities = []

                    # FastAPI accepts a maximum of 1000 records
                    # in one /predict/batch request.
                    chunks = [
                        df.iloc[i:i + BATCH_SIZE]
                        for i in range(
                            0,
                            len(df),
                            BATCH_SIZE,
                        )
                    ]

                    progress = st.progress(0)

                    try:

                        with st.spinner(
                            f"Predicting {len(df):,} patients..."
                        ):

                            for number, chunk in enumerate(chunks):

                                payload = {
                                    "requests": chunk[
                                        required_columns
                                    ].to_dict(
                                        orient="records"
                                    )
                                }

                                response = requests.post(
                                    f"{API_URL}/predict/batch",
                                    json=payload,
                                    timeout=160,
                                )

                                response.raise_for_status()

                                result = response.json()

                                predictions.extend(
                                    result["predictions"]
                                )

                                probabilities.extend(
                                    result["probabilities"]
                                )

                                progress.progress(
                                    (number + 1) / len(chunks)
                                )

                        # -----------------------------------------------------
                        # ADD API RESULTS TO THE ORIGINAL DATA
                        # -----------------------------------------------------

                        df["prediction"] = predictions

                        df["probability"] = probabilities

                        df["is_panic_disorder"] = (
                            df["prediction"] == 1
                        )

                        panic_count = int(
                            df["is_panic_disorder"].sum()
                        )

                        # -----------------------------------------------------
                        # SUMMARY
                        # -----------------------------------------------------

                        st.divider()

                        col1, col2, col3 = st.columns(3)

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

                        # -----------------------------------------------------
                        # RESULTS
                        # -----------------------------------------------------

                        st.subheader("Prediction Results")

                        st.dataframe(
                            df,
                            use_container_width=True,
                        )

                        # -----------------------------------------------------
                        # DOWNLOAD
                        # -----------------------------------------------------

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
                            "The FastAPI request timed out."
                        )

                    except requests.exceptions.HTTPError:

                        st.error(
                            f"API returned an error: "
                            f"{response.status_code}"
                        )

                        try:
                            st.json(response.json())
                        except Exception:
                            st.write(response.text)

                    except Exception as e:

                        st.error(f"Batch prediction error: {e}")

        except Exception as e:

            st.error(
                f"Could not read the CSV file: {e}"
            )


# =============================================================================
# API HEALTH
# =============================================================================

elif mode == "API Health":

    st.header("API Health")

    if st.button("Refresh"):

        st.rerun()

    try:

        # ---------------------------------------------------------------------
        # HEALTH
        # ---------------------------------------------------------------------

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

    except Exception as e:

        st.error(
            f"API error: {e}"
        )

