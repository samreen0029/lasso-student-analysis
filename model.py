import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    # Load dataset
    df = pd.read_csv("student_data.csv")

    # Fix column issues
    df.columns = df.columns.str.strip()
    df = df.dropna()

    # Features and target
    X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours',
            'Previous_Scores', 'Internet_Usage']]
    y = df['Final_Score']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Lasso model
    model = Lasso(alpha=0.5)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    return model, scaler, mse, r2, coefficients
