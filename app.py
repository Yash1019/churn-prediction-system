from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature names
model = joblib.load("model.pkl")
feature_names = joblib.load("features.pkl")   # comes from X_train.columns

# Raw input features (before dummies)
raw_features = ['gender','SeniorCitizen','Partner','Dependents','tenure',
                'PhoneService','MultipleLines','InternetService','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                'MonthlyCharges','TotalCharges']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect raw input
        input_data = {col: request.form[col] for col in raw_features}
        df = pd.DataFrame([input_data])

        # Convert numeric values
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'])
        df['tenure'] = pd.to_numeric(df['tenure'])
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'])
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce").fillna(0)

        # Apply same get_dummies as training
        df = pd.get_dummies(df)

        # Align with training features
        df = df.reindex(columns=feature_names, fill_value=0)

        # Predict
        prediction = model.predict(df)[0]
        result = "❌ Customer is likely to Churn" if prediction == 1 else "✅ Customer will Stay"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
