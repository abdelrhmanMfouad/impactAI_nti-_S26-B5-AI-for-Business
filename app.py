
import json
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Post-GPA Predictor",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "impactAI.pkl"
    )


# ============================================================
# LOAD OPTIONS
# ============================================================

@st.cache_data
def load_options():

    with open(
        "options.json",
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD
# ============================================================

try:

    model = load_model()

    options = load_options()

except Exception as e:

    st.error(
        "❌ Could not load the model or options."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎓 Student Post-GPA Predictor"
)

st.write(
    """
    Predict a student's expected Post-GPA
    using academic, behavioral, and AI-related information.
    """
)

st.divider()


# ============================================================
# GET FEATURE NAMES
# ============================================================

try:

    feature_names = list(
        model.named_steps[
            "preprocessing"
        ].feature_names_in_
    )

except Exception:

    st.error(
        "Could not detect model input features."
    )

    st.stop()


# ============================================================
# CREATE INPUTS
# ============================================================

user_input = {}


st.subheader(
    "📝 Student Information"
)


for feature in feature_names:

    # --------------------------------------------------------
    # CATEGORICAL FEATURE
    # --------------------------------------------------------

    if feature in options:

        selected_value = st.selectbox(

            feature.replace(
                "_",
                " "
            ).title(),

            options[feature],

            index=None,

            placeholder=(
                f"Search or select "
                f"{feature.replace('_', ' ').title()}..."
            )
        )

        user_input[feature] = selected_value


    # --------------------------------------------------------
    # NUMERICAL FEATURE
    # --------------------------------------------------------

    else:

        user_input[feature] = st.number_input(

            feature.replace(
                "_",
                " "
            ).title(),

            value=0.0
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()


predict_button = st.button(
    "🔮 Predict Post-GPA",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CHECK CATEGORICAL VALUES
    # --------------------------------------------------------

    missing = []

    for feature in options:

        if feature in user_input:

            if user_input[feature] is None:

                missing.append(feature)


    if missing:

        st.warning(
            "⚠️ Please select all categorical fields."
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [user_input],
        columns=feature_names
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_df
        )[0]


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Prediction Result"
        )


        st.metric(
            "Predicted Post-GPA",
            f"{prediction:.2f}"
        )


        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        if prediction >= 3.5:

            st.success(
                "🟢 High expected academic performance"
            )

        elif prediction >= 2.5:

            st.warning(
                "🟡 Moderate expected academic performance"
            )

        else:

            st.error(
                "🔴 Low expected academic performance"
            )


        # ----------------------------------------------------
        # INPUT SUMMARY
        # ----------------------------------------------------

        with st.expander(
            "🔎 View entered information"
        ):

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI for Business • Student Post-GPA Prediction"
)

