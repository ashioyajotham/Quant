from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved XGBoost model
xgb = pickle.load(open('jp_morgan_model.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__, static_folder='static')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = request.form.to_dict()

    # Convert the input data to a Pandas DataFrame
    input_df = pd.DataFrame.from_dict(input_data, orient='index').T

    # Convert the input data to the format expected by the model
    input_array = np.array(input_df).astype(np.float32)

    # Make a prediction using the loaded model
    prediction = xgb.predict(input_array)[0]

    # Create a bar plot of the feature importances
    feature_importances = pd.Series(xgb.feature_importances_, index=input_df.columns)
    feature_importances.plot(kind='bar')

    # Add labels to the plot
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')    
    plt.xticks(np.arange(6), ['Open', 'High', 'Low', 'Close', 'Volume', 'Year'], rotation=45)
    plt.tight_layout()

    # Display the plot
    #plt.show()

    # Save the plot to a file
    plt.savefig('static/feature_importances.png')

    # Render the prediction and plot on the results page
    return render_template('result.html', prediction=prediction, plot_url='feature_importances.png')

if __name__ == '__main__':
    app.run(debug=True)