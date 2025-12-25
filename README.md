# Medical Malpractice Claim Predictor

**Description:**  
A machine learning application to predict the likelihood of high-severity medical malpractice claims based on historical US malpractice data. Built with Random Forest and deployed as an interactive Streamlit app.

---

## **Dataset**  
- Source: [US Medical Negligence Dataset]  
- Features include: Amount, Severity, Age, Specialty, Insurance, Gender, Private Attorney, Marital Status.

---

## **Project Workflow**
1. **Data Cleaning & Preprocessing**
   - Missing values handled (numeric → mean, categorical → mode)
   - Categorical features encoded using OneHotEncoder
2. **Feature Engineering**
   - Numeric features scaled
   - Categorical features encoded
3. **Model Training**
   - Random Forest Classifier trained on processed dataset
   - Model and encoder saved using `joblib`
4. **Streamlit App**
   - Users input claim details via GUI
   - Predicts whether a claim is likely high-severity

---

## **How to Run Locally**
1. Clone the repo:
   ```bash
   git clone <repo_url>
   cd medical_malpractice_predictor

2. Install requirements:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py