from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # For Cross-Origin Resource Sharing
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow requests from different origins

# Load the trained model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("fake_news_model")
tokenizer = RobertaTokenizer.from_pretrained("fake_news_model")

# Define the possible labels for the model output
LABELS = ['false', 'true', 'unverified', 'exaggerated', 'biased', 'satirical']

# Route to render the frontend
@app.route('/')
def index():
    return render_template("index.html")

# Route to analyze text and predict if it is fake news or not
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "Please provide text data"})

    text = data['text']
    
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # Perform inference with the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Map the predicted class ID to the label
    predicted_label = LABELS[predicted_class_id]

    # Return the prediction result
    return jsonify({"prediction": predicted_label})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)