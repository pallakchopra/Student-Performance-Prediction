
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #f1f5f9;
}

.subtitle {
    color: #9ca3af;
    font-size: 15px;
}

.result-box {
    padding: 18px;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 20px;
}

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border-radius: 8px;
    background-color: #2563eb;
    color: white;
    border: none;
    padding: 10px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

[data-testid="stMetricValue"] {
    color: #60a5fa;
}

hr {
    border-color: #30363d;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "active_page" not in st.session_state:
    st.session_state.active_page = "Predictor"


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    preprocessor = joblib.load("models/preprocessor.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    random_forest = joblib.load("models/random_forest.pkl")

    return preprocessor, label_encoder, random_forest


try:
    preprocessor, label_encoder, model = load_models()

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --------------------------------------------------
# FEATURE NAME MAPPING
# --------------------------------------------------

feature_names = {
    "G1": "First Period Grade",
    "G2": "Second Period Grade",
    "failures": "Past Class Failures",
    "age": "Student Age",
    "absences": "School Absences",
    "Medu": "Mother's Education Level",
    "Fedu": "Father's Education Level",
    "goout": "Going Out with Friends",
    "studytime": "Weekly Study Time",
    "Walc": "Weekend Alcohol Consumption",
    "Dalc": "Weekday Alcohol Consumption",
    "freetime": "Free Time After School",
    "health": "Current Health Status",
    "famrel": "Family Relationship Quality",
    "traveltime": "Travel Time to School",
    "schoolsup": "Extra School Support",
    "famsup": "Family Educational Support",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery School",
    "higher": "Wants Higher Education",
    "internet": "Internet Access at Home",
    "romantic": "Romantic Relationship",
    "school": "School",
    "sex": "Gender",
    "address": "Area of Residence",
    "famsize": "Family Size",
    "Pstatus": "Parents' Living Status",
    "Mjob": "Mother's Occupation",
    "Fjob": "Father's Occupation",
    "reason": "Reason for Choosing School",
    "guardian": "Student's Guardian"
}

category_values = {
    "GP": "Gabriel Pereira",
    "MS": "Mousinho da Silveira",
    "F": "Female",
    "M": "Male",
    "U": "Urban",
    "R": "Rural",
    "GT3": "More than 3 members",
    "LE3": "3 or fewer members",
    "A": "Apartments / Separate",
    "T": "Together",
    "at_home": "At Home",
    "health": "Healthcare",
    "other": "Other",
    "services": "Services",
    "teacher": "Teacher",
    "course": "Course",
    "home": "Close to Home",
    "reputation": "School Reputation",
    "mother": "Mother",
    "father": "Father",
    "yes": "Yes",
    "no": "No"
}


def readable_feature(name):

    name = str(name)

    if name.startswith("num__"):
        name = name.replace("num__", "")
        return feature_names.get(
            name, name.replace("_", " ").title()
        )

    if name.startswith("cat__"):
        name = name.replace("cat__", "")
        parts = name.split("_", 1)

        if len(parts) == 2:
            attribute, value = parts

            attribute_name = feature_names.get(
                attribute,
                attribute.replace("_", " ").title()
            )

            value_name = category_values.get(
                value,
                value.replace("_", " ").title()
            )

            return f"{attribute_name}: {value_name}"

    return name.replace("_", " ").title()


# --------------------------------------------------
# HEADER AND NAVIGATION
# --------------------------------------------------

header1, header2, header3, header4 = st.columns(
    [4, 1.4, 1.5, 1.5]
)

with header1:
    st.title("🎓 Student Performance Prediction")
    st.markdown(
        '<p class="subtitle">Predict a student\'s academic performance using Machine Learning.</p>',
        unsafe_allow_html=True
    )

with header2:
    st.write("")
    if st.button("🏠 Predictor", use_container_width=True):
        st.session_state.active_page = "Predictor"
        st.rerun()

with header3:
    st.write("")
    if st.button("📊 Model Comparison", use_container_width=True):
        st.session_state.active_page = "Comparison"
        st.rerun()

with header4:
    st.write("")
    if st.button("🎯 Feature Importance", use_container_width=True):
        st.session_state.active_page = "Importance"
        st.rerun()

st.divider()


# ==================================================
# PAGE 1: PREDICTOR
# ==================================================

if st.session_state.active_page == "Predictor":

    st.header("Enter Student Details")

    st.write(
        "Fill in the student's information below to predict their academic performance."
    )

    with st.form("student_form"):

        col1, col2 = st.columns(2)

        with col1:

            school = st.selectbox("School", ["GP", "MS"])

            sex = st.selectbox("Gender", ["F", "M"])

            age = st.slider("Age", 15, 22, 17)

            address = st.selectbox("Area of Residence", ["U", "R"])

            famsize = st.selectbox("Family Size", ["GT3", "LE3"])

            Pstatus = st.selectbox(
                "Parents' Living Status", ["A", "T"]
            )

            Medu = st.slider(
                "Mother's Education Level", 0, 4, 2
            )

            Mjob = st.selectbox(
                "Mother's Occupation",
                ["at_home", "health", "other", "services", "teacher"]
            )

            reason = st.selectbox(
                "Reason for Choosing School",
                ["course", "home", "reputation", "other"]
            )

            traveltime = st.slider(
                "Travel Time to School", 1, 4, 1
            )

            studytime = st.slider(
                "Weekly Study Time", 1, 4, 2
            )

            failures = st.slider(
                "Past Class Failures", 0, 4, 0
            )

            schoolsup = st.selectbox(
                "Extra School Support", ["yes", "no"]
            )

            paid = st.selectbox(
                "Extra Paid Classes", ["yes", "no"]
            )

            nursery = st.selectbox(
                "Attended Nursery School", ["yes", "no"]
            )

            internet = st.selectbox(
                "Internet Access at Home", ["yes", "no"]
            )

        with col2:

            famrel = st.slider(
                "Family Relationship Quality", 1, 5, 4
            )

            Fedu = st.slider(
                "Father's Education Level", 0, 4, 2
            )

            Fjob = st.selectbox(
                "Father's Occupation",
                ["at_home", "health", "other", "services", "teacher"]
            )

            guardian = st.selectbox(
                "Student's Guardian",
                ["mother", "father", "other"]
            )

            activities = st.selectbox(
                "Extracurricular Activities", ["yes", "no"]
            )

            higher = st.selectbox(
                "Wants to Pursue Higher Education", ["yes", "no"]
            )

            romantic = st.selectbox(
                "In a Romantic Relationship", ["yes", "no"]
            )

            famsup = st.selectbox(
                "Family Educational Support", ["yes", "no"]
            )

            absences = st.slider(
                "School Absences", 0, 100, 5
            )

            freetime = st.slider(
                "Free Time After School", 1, 5, 3
            )

            goout = st.slider(
                "Going Out with Friends", 1, 5, 3
            )

            Dalc = st.slider(
                "Weekday Alcohol Consumption", 1, 5, 1
            )

            Walc = st.slider(
                "Weekend Alcohol Consumption", 1, 5, 2
            )

            health = st.slider(
                "Current Health Status", 1, 5, 4
            )

            G1 = st.slider(
                "First Period Grade (0-20)", 0, 20, 10
            )

            G2 = st.slider(
                "Second Period Grade (0-20)", 0, 20, 10
            )

        st.markdown(
            "<div style='height:15px'></div>",
            unsafe_allow_html=True
        )

        left, center, right = st.columns([1, 2, 1])

        with center:
            submitted = st.form_submit_button(
                "🎓 Predict Student Performance",
                use_container_width=True
            )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if submitted:

        input_data = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
            "G1": G1,
            "G2": G2
        }

        input_df = pd.DataFrame([input_data])

        try:

            processed_input = preprocessor.transform(input_df)

            prediction = model.predict(processed_input)

            probabilities = model.predict_proba(
                processed_input
            )[0]

            predicted_category = label_encoder.inverse_transform(
                prediction
            )[0]

            confidence = np.max(probabilities) * 100

            st.session_state["prediction"] = predicted_category
            st.session_state["confidence"] = confidence
            st.session_state["probabilities"] = probabilities

        except Exception as e:
            st.error(f"Prediction error: {e}")


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    if "prediction" in st.session_state:

        st.divider()

        st.header("📊 Prediction Results")

        prediction = st.session_state["prediction"]

        confidence = st.session_state["confidence"]

        probabilities = st.session_state["probabilities"]

        colors = {
            "Excellent": "#14532d",
            "Good": "#1e3a5f",
            "Average": "#78350f",
            "Low": "#7f1d1d"
        }

        st.markdown(
            f"""
            <div class="result-box"
            style="background-color:{colors[prediction]};">
            🎓 {prediction} Performance
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Performance Category Probabilities")

        probability_df = pd.DataFrame({
            "Performance Category": label_encoder.classes_,
            "Probability": probabilities * 100
        })

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=probability_df["Performance Category"],
            y=probability_df["Probability"],
            marker_color="#A78BFA",
            text=probability_df["Probability"].round(2).astype(str) + "%",
            textposition="outside"
        ))

        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Probability (%)",
            xaxis_title="Performance Category",
            yaxis_range=[0, 110],
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Predictions are estimates generated by the trained Random Forest model."
        )


# ==================================================
# PAGE 2: MODEL COMPARISON
# ==================================================

elif st.session_state.active_page == "Comparison":

    st.title("📊 Model Performance Comparison")

    st.write(
        "Compare the performance of the Deep Neural Network "
        "and Random Forest models."
    )

    try:

        comparison = pd.read_csv("model_comparison.csv")

        comparison["Model"] = comparison["Model"].replace({
            "DNN": "Deep Neural Network",
            "Random Forest": "Random Forest"
        })

        metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        for metric in metrics:

            comparison[metric] = pd.to_numeric(
                comparison[metric],
                errors="coerce"
            )

            if comparison[metric].max() > 1:
                comparison[metric] = comparison[metric] / 100

        st.subheader("Performance Metrics")

        display_df = comparison.copy()

        for metric in metrics:
            display_df[metric] = (
                display_df[metric] * 100
            ).map(lambda x: f"{x:.2f}%")

        st.dataframe(
            display_df.set_index("Model"),
            use_container_width=True
        )

        st.subheader("Model Comparison Chart")

        fig = go.Figure()

        colors = {
            "Deep Neural Network": "#A78BFA",
            "Random Forest": "#F472B6"
        }

        for _, row in comparison.iterrows():

            fig.add_trace(go.Bar(
                name=row["Model"],
                x=metrics,
                y=[row[m] * 100 for m in metrics],
                marker_color=colors.get(
                    row["Model"], "#A78BFA"
                ),
                text=[
                    f"{row[m] * 100:.2f}%"
                    for m in metrics
                ],
                textposition="outside"
            ))

        fig.update_layout(
            template="plotly_dark",
            barmode="group",
            title="Model Comparison Across Evaluation Metrics",
            xaxis_title="Evaluation Metrics",
            yaxis_title="Score (%)",
            yaxis_range=[0, 115],
            height=480,
            legend_title="Models"
        )

        st.plotly_chart(fig, use_container_width=True)

        best_model = comparison.loc[
            comparison["Accuracy"].idxmax()
        ]

        st.success(
            f"Best performing model: {best_model['Model']} "
            f"with {best_model['Accuracy'] * 100:.2f}% accuracy."
        )

    except Exception as e:

        st.error(f"Unable to load model comparison: {e}")


# ==================================================
# PAGE 3: FEATURE IMPORTANCE
# ==================================================

elif st.session_state.active_page == "Importance":

    st.title("🎯 Important Factors Affecting Performance")

    st.write(
        "Explore which student attributes have the greatest "
        "influence on academic performance."
    )

    try:

        importance_df = pd.read_csv("feature_importance.csv")

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        importance_df["Readable Feature"] = (
            importance_df["Feature"].apply(readable_feature)
        )

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=True
        )

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Readable Feature"],
            orientation="h",
            marker_color="#A78BFA",
            text=importance_df["Importance"].map(
                lambda x: f"{x:.3f}"
            ),
            textposition="outside"
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Top 10 Important Student Attributes",
            xaxis_title="Importance Score",
            yaxis_title="Student Attributes",
            height=550,
            xaxis_range=[
                0,
                importance_df["Importance"].max() * 1.18
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Higher importance scores indicate that a feature "
            "contributes more to the Random Forest model's predictions. "
            "They do not necessarily indicate causation."
        )

        st.subheader("Feature Importance Values")

        display_importance = importance_df[
            ["Readable Feature", "Importance"]
        ].sort_values(
            by="Importance",
            ascending=False
        )

        display_importance.columns = [
            "Student Attribute",
            "Importance Score"
        ]

        st.dataframe(
            display_importance,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(f"Unable to load feature importance: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#9ca3af;">
        Student Performance Prediction System |
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
