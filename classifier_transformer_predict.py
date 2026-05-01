import torch
import joblib
from transformers import (
	AutoTokenizer, 
    AutoModelForSequenceClassification,
)
import pandas as pd
import re
import datetime


# Configuration

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

NEW_DATA_INPUT_FILE = "Data/New_Articles.xlsx"
MODEL_PATH = "Models/newsletter_transformer_v1"
OUTPUT_FILE = f"Data/New_Articles_Predicted_{timestamp}.xlsx"


# Load models

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")


# Use CPU by default and change to cuda if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

print("Model loaded")
print("Device:", device)


# Define predict function

def predict_text(text):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    label = label_encoder.inverse_transform([pred_idx])[0]

    return label, confidence


# Load dataset 

df = pd.read_excel(NEW_DATA_INPUT_FILE)


# Remove duplicates

df.drop_duplicates(subset='article_url', inplace=True)


# Join columns for prediction

df['Content'] = (df['title'].fillna('') + ' ' + df['tags'].fillna('')).str.strip()
print(df['Content'])


# Remove rows with empty cells

df = df[df['Content'].notna()].copy()


## Transform Content column to lowercase

df['Content'] = df['Content'].fillna('').str.lower()
print(df['Content'])


## Remove lines with Japanese/Chinese/Korean text

def contains_cjk(text):
    if pd.isna(text):
        return False
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

df = df[~df['Content'].apply(contains_cjk)]


# Loop over dataset and classify content

predictions = []
confidences = []

for i, row in df.iterrows():
    try:
        text = row['Content']
        label, conf = predict_text(text)
        predictions.append(label)
        confidences.append(round(conf * 100, 2))
        print(f"{i}: {label} ({conf*100:.2f}%)")
    except Exception as e:
        predictions.append('ERROR')
        confidences.append('ERROR')
        print(f'An error occurred during classification: {e}')
        

# Add predicted data to dataframe

df["Predicted_Subcategory"] = predictions 
df["Confidence"] = confidences


# Sort rows based on categories

df = df.sort_values(by="Predicted_Subcategory")
 
df.to_excel(OUTPUT_FILE, index=False)

print("Finished.")
print("Saved:", OUTPUT_FILE)