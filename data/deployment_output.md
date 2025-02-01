**Deploying the Model as a Web Application**

To deploy the selected model as a web application, we will use the Gradio library. Gradio is an easy-to-use library that allows you to create simple web applications for your machine learning models.

### Required Libraries

We will require the following libraries to deploy the model as a web application:

* `gradio`: For building the web application
* `numpy`: For numerical computations
* `pandas`: For data manipulation
* `sklearn`: For loading the trained model

### Code

Here is an example code snippet that demonstrates how to deploy the selected model as a web application:
```python
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.model_selection import load_model

# Load the trained model
model = load_model('trained_model.joblib')

# Define a function to make predictions
def make_prediction(input_data):
    # Preprocess the input data
    input_data = pd.DataFrame(input_data)
    # Make predictions
    predictions = model.predict(input_data)
    return predictions

# Create a Gradio interface
demo = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.Number(label='Feature 1'),
        gr.Number(label='Feature 2'),
        gr.Number(label='Feature 3'),
        # Add more features as needed
    ],
    outputs=[
        gr.Number(label='Prediction')
    ],
    title='Machine Learning Model Deployment',
    description='Enter input values to get a prediction'
)

# Launch the Gradio interface
if __name__ == '__main__':
    demo.launch()
```
### Explanation

1. We first import the required libraries.
2. We load the trained model using the `load_model` function from Scikit-learn.
3. We define a function `make_prediction` that takes in input data, preprocesses it, and makes predictions using the trained model.
4. We create a Gradio interface using the `Interface` class, specifying the input and output interfaces.
5. We launch the Gradio interface using the `launch` method.

### Example Use Case

Assuming we have a trained model that takes in three features (`Age`, `Income`, and `Education`) and predicts a continuous output (`Credit Score`). We can deploy this model as a web application using the above code snippet.

1. The user enters input values for `Age`, `Income`, and `Education` in the input fields.
2. The `make_prediction` function is called with the input values.
3. The trained model makes a prediction based on the input values.
4. The predicted `Credit Score` is displayed in the output field.

Note: This is a basic example and may need to be modified to accommodate specific requirements, such as data preprocessing, feature engineering, and model selection.