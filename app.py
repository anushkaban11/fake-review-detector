import streamlit as st
import joblib

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Fake Product Review Detector")
st.markdown("**Developed by: Ishmit Singh | 1st Semester Project**")
st.divider()

st.write("This project uses Natural Language Processing (NLP) to detect whether a product review is REAL or FAKE (Bot/Paid).")
st.info("Paste any Amazon / Flipkart review below to test.")

review_text = st.text_area("Enter Review Text:", height=150, placeholder="e.g., This product is amazing, best best buy now 5 stars!!!")

if st.button("Analyze Review", type="primary"):
    if review_text.strip() == "":
        st.warning("Please enter a review first.")
    else:
        try:
            model = joblib.load('fake_review_model.pkl')
            vectorizer = joblib.load('vectorizer.pkl')

            transformed_review = vectorizer.transform([review_text])
            prediction = model.predict(transformed_review)[0]
            probability = model.predict_proba(transformed_review).max()

            st.divider()
            if prediction == 1:
                st.error(f"🚨 FAKE Review Detected! (Confidence: {probability:.2%})")
                st.markdown("**Reason:** This review shows patterns of spam, repetitive words, or overly promotional language.")
            else:
                st.success(f"✅ REAL Review Detected! (Confidence: {probability:.2%})")
                st.markdown("**Reason:** This review appears to be genuine and written by a real customer.")

        except FileNotFoundError:
            st.warning("Model not trained yet. Running in Demo Mode...")
            # Fallback demo logic
            if "best best" in review_text.lower() or "amazing amazing" in review_text.lower() or review_text.count("!") > 3:
                st.error("🚨 Demo Result: FAKE Review Detected!")
            else:
                st.success("✅ Demo Result: REAL Review Detected!")

st.divider()
st.caption("Technologies Used: Python, Scikit-Learn, TF-IDF Vectorizer, Logistic Regression, Streamlit")
