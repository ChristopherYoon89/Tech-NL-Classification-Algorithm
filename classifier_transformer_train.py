import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)  


# Read dataset 

data = pd.read_excel("Data/Dataset_Final_30042026.xlsx")

data = data.dropna(subset=["Content", "Subcategory"])
data = data[data["Subcategory"] != "None"]

print(data["Subcategory"].value_counts())


# Label encoding

label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["Subcategory"])

num_labels = len(label_encoder.classes_)

print("Classes:", list(label_encoder.classes_))


# Split dataset into training and test 

train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


# Convert pandas dataframe to hugging face dataset 

train_ds = Dataset.from_pandas(train_df[["Content", "label"]])
test_ds = Dataset.from_pandas(test_df[["Content", "label"]])


# Tokenize data

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["Content"],
        padding=True,
        truncation=True,
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# Add label for readable predictions

id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}


# Define pretrained model

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    )


# Define function for metrics computation

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# Define training parameters

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True
)


# Initialize trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)


# Train model

trainer.train()


# Test model

predictions = trainer.predict(test_ds)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("Accuracy:", accuracy_score(y_true, y_pred))
print()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# Save model

model.save_pretrained("Models/newsletter_transformer_v1")
tokenizer.save_pretrained("Models/newsletter_transformer_v1")
joblib.dump(label_encoder, "Models/newsletter_label_encoder_v1.pkl")

print("Model saved.")