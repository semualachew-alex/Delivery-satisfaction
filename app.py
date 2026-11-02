"""
Delivery Care Satisfaction Predictor (Streamlit)
------------------------------------------------
Predicts whether a mother is satisfied with delivery care, using the logistic
regression model trained in the APHI machine-learning notebook.

Run:
    pip install -r requirements.txt
    streamlit run app8.py

The trained model bundle is embedded in the notebook's output.
"""

import base64
import io
import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Delivery Care Satisfaction Predictor",
    page_icon="🤰",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_NOTEBOOK = BASE_DIR / "The_last_maternal_satisfaction_pridiction_model.ipynb"
MODEL_MIME_TYPE = "application/vnd.maternal-satisfaction.model"

# Results copied from the notebook (20 % hold-out test set, n = 179).
PERFORMANCE = {
    "Accuracy": 0.978,
    "Precision": 0.983,
    "Recall": 0.983,
    "F1-score": 0.983,
    "Brier score": 0.024,
}

SAT, NOT_SAT = "Satisfied", "Not satisfied"
YES, NO = "Yes", "No"

# Model column -> how it is shown in the app.
# "positive" is the answer that corresponds to the dummy column being 1.
FEATURES = {
    # --- Respectful and supportive care -----------------------------------
    "politenessrecode_satisfied": {
        "label": "Politeness and attitude of staff",
        "group": "Respectful and supportive care",
        "options": (SAT, NOT_SAT),
    },
    "encourageandsupportatdeliveryrecode_satisfied": {
        "label": "Encouragement and support during delivery",
        "group": "Respectful and supportive care",
        "options": (SAT, NOT_SAT),
    },
    "counselingrecode_satisfied": {
        "label": "Counseling services received",
        "group": "Respectful and supportive care",
        "options": (SAT, NOT_SAT),
    },
    "privacyrecode_satisfied": {
        "label": "Visual and auditory privacy",
        "group": "Respectful and supportive care",
        "options": (SAT, NOT_SAT),
    },
    "q113privac_yes": {
        "label": "Were you offered sufficient privacy?",
        "group": "Respectful and supportive care",
        "options": (YES, NO),
    },
    # --- Facility and environment ------------------------------------------
    "availabilityofbedintheward_satisfied": {
        "label": "Availability of beds in the ward",
        "group": "Facility and environment",
        "options": (SAT, NOT_SAT),
    },
    "waitingareacleanessandconfort_satisfied": {
        "label": "Waiting area cleanliness and comfort",
        "group": "Facility and environment",
        "options": (SAT, NOT_SAT),
    },
    "accessandcleanessoftoiletrecode_satisfied": {
        "label": "Toilet access and cleanliness",
        "group": "Facility and environment",
        "options": (SAT, NOT_SAT),
    },
    "overallcleanessrecode_satisfied": {
        "label": "Overall cleanliness of the facility",
        "group": "Facility and environment",
        "options": (SAT, NOT_SAT),
    },
    # --- Timeliness ---------------------------------------------------------
    "waitingtimerecode_satisfied": {
        "label": "Waiting time before being served",
        "group": "Timeliness",
        "options": (SAT, NOT_SAT),
    },
}

# Short explanation shown in the (?) icon next to each question.
HELP = {
    "politenessrecode_satisfied": "Were you satisfied with how politely and respectfully the staff treated you?",
    "encourageandsupportatdeliveryrecode_satisfied": "Were you satisfied with the encouragement and emotional support you received during labour and delivery?",
    "counselingrecode_satisfied": "Were you satisfied with the counseling and health information given to you?",
    "privacyrecode_satisfied": "Were you satisfied with visual and auditory privacy (screens, curtains, closed doors, not being overheard)?",
    "q113privac_yes": "Did the facility offer you enough privacy during care?",
    "availabilityofbedintheward_satisfied": "Were you satisfied with the availability of a bed in the ward?",
    "waitingareacleanessandconfort_satisfied": "Were you satisfied with how clean and comfortable the waiting area was?",
    "accessandcleanessoftoiletrecode_satisfied": "Were you satisfied with access to a toilet and how clean it was?",
    "overallcleanessrecode_satisfied": "Were you satisfied with the overall cleanliness of the facility?",
    "waitingtimerecode_satisfied": "Were you satisfied with how long you waited before being served?",
}

GROUP_ORDER = [
    "Respectful and supportive care",
    "Facility and environment",
    "Timeliness",
]


def pretty_name(col: str) -> str:
    """Friendly label for a model column (falls back to the raw column name)."""
    return FEATURES.get(col, {}).get("label", col)


# --------------------------------------------------------------------------- #
# Loading the trained model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    if not MODEL_NOTEBOOK.is_file():
        raise FileNotFoundError(f"Model notebook not found: {MODEL_NOTEBOOK.name}")

    with MODEL_NOTEBOOK.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    encoded_bundle = next(
        (
            output.get("data", {}).get(MODEL_MIME_TYPE)
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if output.get("data", {}).get(MODEL_MIME_TYPE)
        ),
        None,
    )
    if encoded_bundle is None:
        raise ValueError(
            "The notebook does not contain an embedded model. "
            "Run the final model export cell and save the notebook."
        )

    bundle = joblib.load(io.BytesIO(base64.b64decode(encoded_bundle)))
    columns = pd.Index(bundle["columns"])
    selected = list(columns[bundle["selector"].get_support()])
    return bundle["scaler"], bundle["selector"], bundle["model"], columns, selected


try:
    assets = load_assets()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

scaler, selector, model, all_columns, selected_features = assets

# Only ask about features the model actually uses (and that we have labels for).
active = [c for c in selected_features if c in FEATURES]
unknown = [c for c in selected_features if c not in FEATURES]
if unknown:
    st.error(
        "The saved model uses features this app has no question for: "
        + ", ".join(unknown)
        + ". Add them to `FEATURES` in app8.py."
    )
    st.stop()


# Default answer for every question: the most common response in the survey
# (for all ten features this is "Satisfied" / "Yes", i.e. the first option).
DEFAULTS = {c: FEATURES[c]["options"][0] for c in active}


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict(answers: dict):
    """answers: {column: True/False}. Returns probability and per-feature contributions."""
    # The scaler was fitted on all encoded columns, so rebuild a full row. Columns the
    # selector drops are left at 0 - they are removed before reaching the model.
    row = pd.DataFrame(0.0, index=[0], columns=all_columns)
    for col, value in answers.items():
        row.loc[0, col] = float(value)

    selected = selector.transform(scaler.transform(row))
    probability = float(model.predict_proba(selected)[0, 1])

    # Logistic regression is additive on the log-odds scale: coefficient x scaled value.
    contributions = pd.DataFrame(
        {
            "Factor": [pretty_name(c) for c in selected_features],
            "Contribution": model.coef_[0] * selected[0],
        }
    )
    return probability, contributions, float(model.intercept_[0])


# --------------------------------------------------------------------------- #
# Session-state helpers (quick-fill / reset)
# --------------------------------------------------------------------------- #
def key_of(col: str) -> str:
    return f"in_{col}"


def fill_all(satisfied: bool):
    for col in active:
        pos, neg = FEATURES[col]["options"]
        st.session_state[key_of(col)] = pos if satisfied else neg


def reset_all():
    """Back to the default answers (the most common answer in the survey)."""
    for col in active:
        st.session_state[key_of(col)] = DEFAULTS[col]
    st.session_state.pop("result", None)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("About")
    st.write(
        "Machine-learning model (logistic regression) built on a survey of mothers "
        "who delivered in health facilities in Bahir Dar city, Northwest Ethiopia."
    )
    st.subheader("Model performance")
    st.caption("Hold-out test set (n = 179)")
    perf = pd.DataFrame({"Metric": list(PERFORMANCE), "Value": list(PERFORMANCE.values())})
    st.dataframe(
        perf,
        hide_index=True,
        column_config={"Value": st.column_config.NumberColumn(format="%.3f")},
    )
    st.subheader("Please note")
    st.caption(
        "This tool is for research and education. It summarises patterns in one survey "
        "and is not a clinical or administrative decision tool. The inputs are themselves "
        "satisfaction ratings, so the model mainly shows how service-quality ratings "
        "relate to overall satisfaction."
    )

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🤰 Delivery Care Satisfaction Predictor")
st.write(
    "Rate each aspect of the delivery care service. The model estimates the chance "
    "that the mother was **satisfied overall**."
)

c1, c2, c3 = st.columns([1, 1, 1])
c1.button("✅ Fill all satisfied", on_click=fill_all, args=(True,))
c2.button("❌ Fill all not satisfied", on_click=fill_all, args=(False,))
c3.button("↺ Reset to defaults", on_click=reset_all)

with st.form("prediction_form"):
    # Flat two-column layout: first half of the questions on the left, rest on the right.
    ordered = [c for g in GROUP_ORDER for c in active if FEATURES[c]["group"] == g]
    half = (len(ordered) + 1) // 2
    left_col, right_col = st.columns(2, gap="large")
    for column, items in ((left_col, ordered[:half]), (right_col, ordered[half:])):
        with column:
            for col in items:
                meta = FEATURES[col]
                st.selectbox(
                    meta["label"],
                    options=list(meta["options"]),
                    index=list(meta["options"]).index(DEFAULTS[col]),
                    help=HELP.get(col),
                    key=key_of(col),
                )
    try:  # full-width button (newer Streamlit)
        submitted = st.form_submit_button("▶ Predict Satisfaction", type="primary", width="stretch")
    except TypeError:  # older Streamlit
        submitted = st.form_submit_button("▶ Predict Satisfaction", type="primary", use_container_width=True)

if submitted:
    answers = {c: st.session_state.get(key_of(c)) for c in active}
    missing = [pretty_name(c) for c, v in answers.items() if v is None]
    if missing:
        st.session_state.pop("result", None)
        st.warning("Please answer every question first. Missing: " + "; ".join(missing))
    else:
        encoded = {c: answers[c] == FEATURES[c]["options"][0] for c in active}
        st.session_state["result"] = predict(encoded)

# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
if "result" in st.session_state:
    probability, contributions, intercept = st.session_state["result"]

    st.divider()
    st.subheader("Result")

    left, right = st.columns([1, 2])
    left.metric("Probability of satisfaction", f"{probability:.1%}")
    with right:
        if probability >= 0.5:
            st.success("**Predicted outcome: Satisfied**")
        else:
            st.warning("**Predicted outcome: Not satisfied**")
        st.progress(probability)
        st.caption("A probability of 50 % or more is classified as satisfied.")

    st.markdown("#### What drove this prediction?")
    st.caption(
        "Each bar shows how much an answer pushed the result toward **satisfied** "
        "(green, right) or **not satisfied** (red, left). Longer bars mean a bigger effect."
    )
    plot_df = contributions.assign(
        Contribution=contributions["Contribution"].round(3),
        Direction=lambda d: d["Contribution"].map(lambda v: "Toward satisfied" if v >= 0 else "Toward not satisfied"),
    )
    order = plot_df.reindex(plot_df["Contribution"].abs().sort_values(ascending=False).index)["Factor"].tolist()
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Effect on log-odds of satisfaction"),
            y=alt.Y("Factor:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Toward satisfied", "Toward not satisfied"],
                    range=["#2E7D32", "#C62828"],
                ),
                legend=None,
            ),
            tooltip=["Factor", "Contribution", "Direction"],
        )
        .properties(height=34 * len(plot_df) + 20)
    )
    st.altair_chart(chart)

    with st.expander("Technical details"):
        st.write(
            f"Model intercept (log-odds): **{intercept:.3f}**. "
            f"Sum of contributions: **{contributions['Contribution'].sum():.3f}**. "
            f"Intercept + contributions = log-odds; probability = 1 / (1 + e^-log-odds)."
        )
        st.dataframe(
            contributions.sort_values("Contribution", key=abs, ascending=False),
            hide_index=True,
        )
