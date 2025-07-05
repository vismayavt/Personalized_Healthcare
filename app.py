import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pipeline (preprocessing + model)
with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Page configuration
st.set_page_config(page_title="🩺 Healthcare Recommender", layout="centered")
st.title("🩺 Personalized Healthcare Recommendation System")
st.markdown("Enter your health information to receive a personalized medical recommendation.")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    billing_amount = st.number_input("Billing Amount", min_value=0.0, value=1000.0)
    room_number = st.number_input("Room Number", min_value=0, value=101)
    gender = st.selectbox("Gender", ["Male", "Female"])
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
    insurance = st.selectbox("Insurance Provider", ["ProviderA", "ProviderB", "Medicare", "Aetna", "Blue Cross"])
    submitted = st.form_submit_button("Get Recommendation")

# Medical condition recommendations dictionary
condition_recommendations = {
    "Cancer": {
        "summary": "⚠️ High-risk condition detected. Immediate medical attention is advised.",
        "actions": [
            "🔬 Schedule oncology consultation immediately.",
            "🍎 Maintain a nutrient-rich, cancer-supportive diet.",
            "💉 Follow prescribed chemotherapy or radiation schedules strictly."
        ]
    },
    "Diabetes": {
        "summary": "⚠️ Diabetes detected. Consistent management is crucial.",
        "actions": [
            "🍚 Follow a low-sugar, high-fiber diet.",
            "💊 Monitor blood sugar regularly and take prescribed medication.",
            "🏃 Engage in daily physical activity."
        ]
    },
    "Obesity": {
        "summary": "⚠️ Obesity risk identified. Lifestyle modifications are recommended.",
        "actions": [
            "🥗 Shift to a low-calorie, nutrient-dense diet.",
            "🧘 Join a supervised fitness or weight loss program.",
            "📆 Schedule regular checkups with a nutritionist."
        ]
    },
    "Hypertension": {
        "summary": "⚠️ High blood pressure detected. Monitoring and medication advised.",
         "actions": [
            "🧂 Reduce salt intake significantly.",
            "💊 Take antihypertensive medications as prescribed.",
            "🧘 Practice stress management and regular exercise."
        ]
    },
    "Heart Disease": {
        "summary": "⚠️ Signs of cardiovascular risk detected.",
        "actions": [
            "🏥 Schedule an appointment with a cardiologist.",
            "🍇 Follow a heart-healthy diet (low cholesterol, low fat).",
            "🚭 Avoid smoking and alcohol."
        ]
    },
    "Healthy": {
        "summary": "✅ No critical health concerns detected. Keep up the good work!",
        "actions": [
            "✔️ Maintain routine health checkups every 6–12 months.",
            "🥦 Eat a balanced diet and stay hydrated.",
            "🏃‍♂️ Continue regular exercise and avoid stress."
        ]
    },
    "Arthritis": {
        "summary": "🦴 Arthritis detected. Joint care is essential.",
        "actions": [
            "💊 Follow prescribed anti-inflammatory medications.",
            "🧘 Practice joint-friendly exercises like yoga or swimming.",
            "🪑 Use ergonomic furniture to reduce strain on joints."
        ]
    }
}

# On form submit
if submitted:
    # Construct patient input
    patient = {
        "Age": age,
        "Billing Amount": billing_amount,
        "Room Number": room_number,
        "Gender": gender,
        "Blood Type": blood_type,
        "Admission Type": admission_type,
        "Insurance Provider": insurance
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([patient])

    # Add derived feature
    input_df['health_index'] = input_df['Age'] / (input_df['Billing Amount'] + 1)

    # Predict condition
    prediction = pipeline.predict(input_df)[0]

    # Show result
    st.subheader("🎯 Predicted Medical Condition")
    st.success(f"🩺 {prediction}")

    # Show tailored recommendations
    if prediction in condition_recommendations:
        rec = condition_recommendations[prediction]
        st.markdown(f"**{rec['summary']}**")
        st.markdown("### Recommended Actions:")
        for tip in rec["actions"]:
            st.markdown(f"- {tip}")
    else:
        st.info("📝 No specific recommendations found for this condition. Please consult a physician.")
