import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    return joblib.load("model_rf.joblib")

model = load_model()

st.set_page_config(page_title="Lung Cancer Risk Predictor", page_icon="🫁", layout="wide")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Lung Cancer Risk Predictor**  
Built with Streamlit + Random Forest.  
Dataset: Kaggle (Lung Cancer Prediction).  

⚠️ *Disclaimer: This is a demo ML app, not medical advice.*
""")

st.title("🫁 Lung Cancer Risk Predictor")
st.write("Fill in the details below to estimate lung cancer risk level.")

# -----------------------
# Input Sections
# -----------------------

with st.expander("👤 Demographics"):
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

with st.expander("🏠 Lifestyle Factors"):
    air_pollution = st.slider("Air Pollution (1-9)", 1, 9, 5)
    alcohol_use = st.slider("Alcohol use (1-9)", 1, 9, 3)
    dust_allergy = st.slider("Dust Allergy (1-9)", 1, 9, 3)
    occupational_hazards = st.slider("Occupational Hazards (1-9)", 1, 9, 3)
    balanced_diet = st.slider("Balanced Diet (1-9)", 1, 9, 3)
    obesity = st.slider("Obesity (1-9)", 1, 9, 3)
    smoking = st.slider("Smoking (1-9)", 1, 9, 3)
    passive_smoker = st.slider("Passive Smoker (1-9)", 1, 9, 3)
    snoring = st.slider("Snoring (1-9)", 1, 9, 3)

with st.expander("🩺 Medical History"):
    genetic_risk = st.slider("Genetic Risk (1-9)", 1, 9, 3)
    chronic_lung = st.slider("Chronic Lung Disease (1-9)", 1, 9, 3)

with st.expander("⚠️ Symptoms"):
    chest_pain = st.slider("Chest Pain (1-9)", 1, 9, 3)
    coughing_blood = st.slider("Coughing of Blood (1-9)", 1, 9, 3)
    fatigue = st.slider("Fatigue (1-9)", 1, 9, 3)
    weight_loss = st.slider("Weight Loss (1-9)", 1, 9, 3)
    shortness_breath = st.slider("Shortness of Breath (1-9)", 1, 9, 3)
    wheezing = st.slider("Wheezing (1-9)", 1, 9, 3)
    swallowing_diff = st.slider("Swallowing Difficulty (1-9)", 1, 9, 3)
    clubbing = st.slider("Clubbing of Finger Nails (1-9)", 1, 9, 3)
    frequent_cold = st.slider("Frequent Cold (1-9)", 1, 9, 3)
    dry_cough = st.slider("Dry Cough (1-9)", 1, 9, 3)

# -----------------------
# Build input DataFrame
# -----------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Air Pollution": air_pollution,
    "Alcohol use": alcohol_use,
    "Dust Allergy": dust_allergy,
    "OccuPational Hazards": occupational_hazards,
    "Genetic Risk": genetic_risk,
    "chronic Lung Disease": chronic_lung,
    "Balanced Diet": balanced_diet,
    "Obesity": obesity,
    "Smoking": smoking,
    "Passive Smoker": passive_smoker,
    "Chest Pain": chest_pain,
    "Coughing of Blood": coughing_blood,
    "Fatigue": fatigue,
    "Weight Loss": weight_loss,
    "Shortness of Breath": shortness_breath,
    "Wheezing": wheezing,
    "Swallowing Difficulty": swallowing_diff,
    "Clubbing of Finger Nails": clubbing,
    "Frequent Cold": frequent_cold,
    "Dry Cough": dry_cough,
    "Snoring": snoring
}])

# -----------------------
# Prediction
# -----------------------
if st.button("🔮 Predict"):
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    # Color-coded result
    if pred == "Low":
        st.success(f"🟢 Predicted Risk Level: {pred}")
    elif pred == "Medium":
        st.warning(f"🟠 Predicted Risk Level: {pred}")
    else:
        st.error(f"🔴 Predicted Risk Level: {pred}")

    # Show probabilities
    st.subheader("Prediction Confidence")
    prob_df = pd.DataFrame({"Level": model.classes_, "Probability": probs})
    st.bar_chart(prob_df.set_index("Level"))

    # Feature importance
    st.subheader("Top 10 Important Features")
    clf = model.named_steps["clf"]
    feat_names = model.named_steps["prep"].get_feature_names_out()
    importances = clf.feature_importances_

    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    imp_df.plot(kind="barh", x="feature", y="importance", ax=ax, legend=False)
    ax.invert_yaxis()
    st.pyplot(fig)