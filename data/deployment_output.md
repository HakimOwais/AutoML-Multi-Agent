### Dataset Loading and Preprocessing

To load and preprocess the given dataset, we'll use the following Python code:

```python
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset from the given context
data = [
    '{"age": 67, "sex": 0, "cp": 2, "trestbps": 115, "chol": 564, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 1.6, "slope": 1, "ca": 0, "thal": 3, "target": 1}',
    '{"age": 67, "sex": 0, "cp": 2, "trestbps": 115, "chol": 564, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 1.6, "slope": 1, "ca": 0, "thal": 3, "target": 1}',
    '{"age": 67, "sex": 0, "cp": 2, "trestbps": 115, "chol": 564, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 1.6, "slope": 1, "ca": 0, "thal": 3, "target": 1}',
    '{"age": 65, "sex": 0, "cp": 2, "trestbps": 155, "chol": 269, "fbs": 0, "restecg": 1, "thalach": 148, "exang": 0, "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 2, "target": 1}',
    '{"age": 65, "sex": 0, "cp": 2, "trestbps": 155, "chol": 269, "fbs": 0, "restecg": 1, "thalach": 148, "exang": 0, "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 2, "target": 1}',
]

# Parse the JSON data
json_data = [json.loads(d) for d in data]

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(json_data)

# Define the features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Model Loading, Optimization, Training, and Deployment

For this example, we'll use a simple machine learning model, `LogisticRegression`. We'll optimize the model using a grid search over the hyperparameter space.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the model and hyperparameter space
model = LogisticRegression(max_iter=1000)
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
}

# Perform grid search to optimize the model
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model and its performance on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print("Best Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

### Model Preparation for Deployment

To prepare the model for deployment, we'll save it to a file using `joblib`.

```python
import joblib

# Save the best model to a file
joblib.dump(best_model, 'best_model.joblib')
```

### Building a Web Application Demo using Gradio

We'll build a simple web application demo using Gradio to test the deployed model.

```python
import gradio as gr

# Load the saved model
loaded_model = joblib.load('best_model.joblib')

# Define the function to predict the target
def predict_target(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create a DataFrame with the input features
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal],
    })
    
    # Scale the input features
    input_scaled = scaler.transform(input_df)
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_scaled)
    
    return prediction[0]

# Create the Gradio application
demo = gr.Interface(
    fn=predict_target,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex"),
        gr.Number(label="Chest Pain Type (cp)"),
        gr.Number(label="Resting Blood Pressure (trestbps)"),
        gr.Number(label="Cholesterol (chol)"),
        gr.Number(label="Fasting Blood Sugar (fbs)"),
        gr.Number(label="Resting Electrocardiographic Results (restecg)"),
        gr.Number(label="Maximum Heart Rate Achieved (thalach)"),
        gr.Number(label="Exercise Induced Angina (exang)"),
        gr.Number(label="ST Depression Induced by Exercise Relative to Rest (oldpeak)"),
        gr.Number(label="Slope of the Peak Exercise ST Segment (slope)"),
        gr.Number(label="Number of Major Vessels (0-3) Colored by Fluoroscopy (ca)"),
        gr.Number(label="Thalassemia (thal)"),
    ],
    outputs=gr.Number(label="Target"),
    title="Heart Disease Prediction Model",
)

# Launch the Gradio application
demo.launch()
```

### Model Evaluation

To evaluate the model, we'll use the `accuracy_score`, `classification_report`, and `confusion_matrix` functions from Scikit-learn.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
print("Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```