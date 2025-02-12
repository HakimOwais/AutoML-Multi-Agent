### Deploying the Model as a Web Application

To deploy the model as a web application, we'll use Gradio, a Python library that allows us to create simple and shareable web-based interfaces for our models.

#### Step 1: Import Necessary Libraries and Load the Model

First, we need to import the necessary libraries and load the trained model.

```python
import gradio as gr
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = [
    '{"age": 70, "sex": 1, "cp": 2, "trestbps": 160, "chol": 269, "fbs": 0, "restecg": 1, "thalach": 112, "exang": 1, "oldpeak": 2.9, "slope": 1, "ca": 1, "thal": 3, "target": 0}',
    '{"age": 70, "sex": 1, "cp": 2, "trestbps": 160, "chol": 269, "fbs": 0, "restecg": 1, "thalach": 112, "exang": 1, "oldpeak": 2.9, "slope": 1, "ca": 1, "thal": 3, "target": 0}',
    '{"age": 70, "sex": 1, "cp": 2, "trestbps": 160, "chol": 269, "fbs": 0, "restecg": 1, "thalach": 112, "exang": 1, "oldpeak": 2.9, "slope": 1, "ca": 1, "thal": 3, "target": 0}',
    '{"age": 57, "sex": 1, "cp": 1, "trestbps": 124, "chol": 261, "fbs": 0, "restecg": 1, "thalach": 141, "exang": 0, "oldpeak": 0.3, "slope": 2, "ca": 0, "thal": 3, "target": 0}',
    '{"age": 57, "sex": 1, "cp": 1, "trestbps": 124, "chol": 261, "fbs": 0, "restecg": 1, "thalach": 141, "exang": 0, "oldpeak": 0.3, "slope": 2, "ca": 0, "thal": 3, "target": 0}',
]

data = [json.loads(item) for item in dataset]
df = pd.DataFrame(data)

# Define features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
```

#### Step 2: Create a Demo Web Application

Next, we create a simple web application using Gradio that allows users to input their values and get predictions from the model.

```python
# Create a Gradio interface
def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create a dataframe with the input values
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal],
    })

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Get the prediction from the model
    prediction = model.predict(input_scaled)

    # Return the prediction
    return "The predicted target is: " + str(prediction[0])

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Sex"),
        gr.Number(label="Chest Pain (cp)"),
        gr.Number(label="Resting Blood Pressure (trestbps)"),
        gr.Number(label="Serum Cholesterol (chol)"),
        gr.Number(label="Fasting Blood Sugar (fbs)"),
        gr.Number(label="Resting Electrocardiogram (restecg)"),
        gr.Number(label="Maximum Heart Rate Achieved (thalach)"),
        gr.Number(label="Exercise Induced Angina (exang)"),
        gr.Number(label="ST Depression Induced by Exercise Relative to Rest (oldpeak)"),
        gr.Number(label="Slope of the Peak Exercise ST Segment (slope)"),
        gr.Number(label="Number of Major Vessels (ca)"),
        gr.Number(label="Thalassemia (thal)"),
    ],
    outputs=[gr.Textbox(label="Prediction")],
    title="Heart Disease Prediction",
    description="Enter your values to get a prediction",
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
```

To run the web application, save this code in a file (e.g., `app.py`) and run it with `python app.py`. Then, open a web browser and navigate to `http://localhost:7860/` to use the application.