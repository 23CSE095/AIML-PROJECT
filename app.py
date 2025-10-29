import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------------------
# Load the trained model
# -------------------------------
model = tf.keras.models.load_model("skin_cancer_metadata_model.h5")

st.title(" Skin Cancer Prediction (Metadata-based)")
st.write("This app predicts whether a lesion is **Benign or Malignant** using metadata only (no images).")

# -------------------------------
# User Input Section
# -------------------------------
st.header("Enter Patient & Lesion Details")

age = st.number_input("Age", min_value=0, max_value=120, value=40)
gender = st.selectbox("Gender", ["male", "female"])
lesion_type = st.selectbox("Lesion Type", ["melanocytic", "non_melanocytic"])
diameter_1 = st.number_input("Diameter 1 (mm)", min_value=0.0, max_value=50.0, value=10.0)
diameter_2 = st.number_input("Diameter 2 (mm)", min_value=0.0, max_value=50.0, value=8.0)

# -------------------------------
# Preprocessing (same as training)
# -------------------------------
input_data = pd.DataFrame({
    'age': [age],
    'gender_male': [1 if gender == "male" else 0],
    'lesion_type_melanocytic': [1 if lesion_type == "melanocytic" else 0],
    'diameter_1': [diameter_1],
    'diameter_2': [diameter_2]
})

# Convert to numpy array
X_input = np.array(input_data)

# -------------------------------
# Predict
# -------------------------------
if st.button(" Predict"):
    prediction = model.predict(X_input)
    class_names = ["Benign", "Malignant"]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[0]

    st.subheader(f" Prediction: **{predicted_class}**")
    st.write("Confidence Scores:")

    # -------------------------------
    # Chart visualization
    # -------------------------------
    fig, ax = plt.subplots()
    ax.bar(class_names, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

    if predicted_class == "Benign":
        st.success("The lesion is predicted to be Benign.")
    else:
        st.error("The lesion is predicted to be Malignant.")

st.markdown("---")
st.caption("Model based on metadata only — trained with age, gender, lesion type, and lesion diameters.")
