from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

# Load data and preprocess
csv_file = "./email.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Calculate priors
total = len(df)
yes_count = len(df[df['Category'] == 'spam'])
no_count = len(df[df['Category'] == 'ham'])

prior_yes = yes_count / total
prior_no = no_count / total

def likelihood(word, target_value):
    word_count = sum(df['Message'].str.contains(word) & (df["Category"] == target_value))
    target_total = sum(df["Category"] == target_value)
    return (word_count + 1) / (target_total + 2)

def predict(message):
    message = re.sub(r'\W+', ' ', message).lower()
    words = message.split()
    likelihood_yes = [likelihood(word, "spam") for word in words]
    likelihood_no = [likelihood(word, "ham") for word in words]

    v_yes = prior_yes * np.prod(likelihood_yes)
    v_no = prior_no * np.prod(likelihood_no)

    if v_yes + v_no == 0:
        return "Both probabilities are zero due to insufficient data"
    else:
        normalized_yes = v_yes / (v_yes + v_no)
        normalized_no = v_no / (v_yes + v_no)

    if normalized_yes > normalized_no:
        return "spam"
    else:
        return "ham"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    message = request.form['message']
    print("Received message:", message)  # Debug line
    result = predict(message)
    print("Prediction result:", result)  # Debug line
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
