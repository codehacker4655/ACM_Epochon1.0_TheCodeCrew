from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the model and tokenizer from the saved directory
model = RobertaForSequenceClassification.from_pretrained("fake_news_model")
tokenizer = RobertaTokenizer.from_pretrained("fake_news_model")

# Function to predict if a statement is fake or real
def predict_fake_news(statement):
    # Tokenize the input statement
    inputs = tokenizer(statement, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Get model predictions
    with torch.no_grad():  # Turn off gradient calculation for inference
        outputs = model(**inputs)
    
    # Get the logits (raw predictions)
    logits = outputs.logits
    
    # Convert logits to probabilities (optional)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted label (index of the max probability)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    
    # Map the predicted label to a human-readable output (for binary classification: 0 = Real, 1 = Fake)
    label_map = {0: "Real", 1: "Fake"}
    
    # Return the predicted label and its probability
    return label_map[predicted_label], probabilities[0][predicted_label].item()

# Example usage
statement = "The moon landing was a hoax."
prediction, confidence = predict_fake_news(statement)
print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
