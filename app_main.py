import streamlit as st
import pickle

# Load trained sentiment analysis model
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the trained TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit app UI
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("📝 Sentiment Analysis Dashboard")
st.markdown("Enter a customer review below to analyze its sentiment (Positive or Negative).")

# Input text box
user_input = st.text_area("Enter your review text:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        # Vectorize input text
        input_vector = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = round(max(proba) * 100, 2)

        # Display result
        if prediction == 1:
            st.success(f"✅ Sentiment: **Positive** 😊 (Confidence: {confidence}%)")
        else:
            st.error(f"❌ Sentiment: **Negative** 😞 (Confidence: {confidence}%)")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Sentiment Analysis using Logistic Regression + TF-IDF")
