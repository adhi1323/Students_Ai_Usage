import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="AI Usage Predictor", layout="wide")

st.markdown(
    """
    <style>
    /* Make Streamlit app background transparent */
    .stApp {
        background: transparent;
    }

    /* Blurred background image div */
    .background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("https://images.unsplash.com/photo-1677442136019-21780ecad995");
        background-size: cover;
        background-position: center;
        filter: blur(2px);
        z-index: -1; /* behind everything */
    }

    /* Optional: make your content readable with slight overlay */
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.2); /* light semi-transparent overlay */
        z-index: 0; /* behind content, above background */
    }
    </style>

    <div class="background"></div>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='white-space: nowrap;'>🎓 Student AI Usage Prediction App</h1>", unsafe_allow_html=True)

# Load model and encoders
model = joblib.load("best_ai_usage_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Title
st.markdown("### Enter student details below")
st.markdown("---")

# Create two columns
col1, col2 = st.columns(2)

# LEFT COLUMN → Inputs
with col1:
    st.subheader("📥 Student Inputs")

    grades_before_ai = st.number_input("Grades Before AI", min_value=0, max_value=100)

    study_hours_per_day = st.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0)

    daily_screen_time_hours = st.number_input("Daily Screen Time (hours)", min_value=0.0, max_value=24.0)

    education_level = st.selectbox(
        "Education Level",
        label_encoders['education_level'].classes_
    )

    ai_tools_used = st.selectbox(
        "AI Tool Used",
        label_encoders['ai_tools_used'].classes_
    )

    purpose_of_ai = st.selectbox(
        "Purpose of AI",
        label_encoders['purpose_of_ai'].classes_
    )

# Encode inputs
education_level_encoded = label_encoders['education_level'].transform([education_level])[0]
ai_tools_used_encoded = label_encoders['ai_tools_used'].transform([ai_tools_used])[0]
purpose_of_ai_encoded = label_encoders['purpose_of_ai'].transform([purpose_of_ai])[0]

input_data = pd.DataFrame({
    'grades_before_ai': [grades_before_ai],
    'study_hours_per_day': [study_hours_per_day],
    'daily_screen_time_hours': [daily_screen_time_hours],
    'education_level_encoded': [education_level_encoded],
    'ai_tools_used_encoded': [ai_tools_used_encoded],
    'purpose_of_ai_encoded': [purpose_of_ai_encoded]
})

# RIGHT COLUMN → Prediction
with col2:

    if st.button("Predict AI Usage"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        result = target_encoder.inverse_transform(prediction)

        confidence = prediction_proba.max() * 100

        st.success(f"🎯 Prediction: {result[0]}")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        st.subheader("📋 Input Summary")
        st.write(input_data)
