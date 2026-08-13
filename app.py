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
# NUMERIC SCALE CONFIG
# ============================================================
# Maps keywords found in a feature name to (min, max, step, default).
# The first keyword that matches the feature name wins, so put more
# specific keywords first. Anything that matches nothing falls back
# to DEFAULT_SCALE.

SCALE_RULES = [
    # keyword                 min     max     step    default
    ("gpa",                   0.0,    4.0,    0.01,   2.5),
    ("age",                   14,     60,     1,      18),
    ("hour",                  0,      24,     1,      6),
    ("attendance",            0,      100,    1,      75),
    ("percent",               0,      100,    1,      50),
    ("score",                 0,      100,    1,      50),
    ("rating",                1,      5,      1,      3),
    ("year",                  1,      6,      1,      1),
    ("semester",              1,      12,     1,      1),
    ("credit",                0,      30,     1,      15),
    ("count",                 0,      20,     1,      0),
    ("number",                0,      20,     1,      0),
]

DEFAULT_SCALE = (0, 100, 1, 50)


def get_scale(feature_name: str):
    """
    Return (min_value, max_value, step, default_value) for a given
    numeric feature, based on keyword matching against its name.
    """

    lowered = feature_name.lower()

    for keyword, lo, hi, step, default in SCALE_RULES:

        if keyword in lowered:

            return lo, hi, step, default

    return DEFAULT_SCALE


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
    # NUMERICAL FEATURE (now a scaled slider)
    # --------------------------------------------------------

    else:

        lo, hi, step, default = get_scale(feature)

        user_input[feature] = st.slider(

            feature.replace(
                "_",
                " "
            ).title(),

            min_value=lo,
            max_value=hi,
            value=default,
            step=step
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
