import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning messages

import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 inference.py test_reviews.csv test_labels_pred.csv")
        return

    # Get input and output file paths
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Load your fine-tuned model and tokenizer
    model_directory = "./model"
    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # Load the test data
    test_data = pd.read_csv(input_file_path)

    # Tokenize the test text data
    input_ids = []
    attention_masks = []

    for text in test_data['text']:
        encoding = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])

    # Make predictions
    model.eval()
    predictions = []

    for input_id, attention_mask in zip(input_ids, attention_masks):
        with torch.no_grad():
            outputs = model(input_id, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()
            # Map numeric prediction to "Positive" or "Negative"
            if predicted_label == 1:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
            predictions.append(sentiment)

    # Create a new DataFrame with 'id' and 'sentiment' columns
    results_df = pd.DataFrame({'id': test_data['id'], 'sentiment': predictions})

    # Save the results to the output file
    results_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()