from utils.data_reading import read_jsonl_file, load_data_for_expert
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup,  RobertaForSequenceClassification, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

data_path = "/home/liuhui/unify/data"
dataset_name_C = "Constraint"
dataset_name_G = "GOSSIPCOP"
dataset_name_P =  "POLITIFACT"
dataset_name_L = "LIAR-PLUS"
# dataload
# "ID": ID, "MESSAGE": MESSAGE,  "EVIDENCE": EVIDENCE, "label": label
# dataset = {"test": testset, "train": trainset, "val": valset}
C_dataset, _ = load_data_for_expert(os.path.join(data_path, dataset_name_C), dataset_name_C, mode="binary", gq_file=None, sq_file=None, evo_file=None, evo_flag=False)
G_dataset, _ = load_data_for_expert(os.path.join(data_path, dataset_name_G), dataset_name_G, mode="binary", gq_file=None, sq_file=None, evo_file=None, evo_flag=False)
P_dataset, _ = load_data_for_expert(os.path.join(data_path, dataset_name_P), dataset_name_P, mode="binary", gq_file=None, sq_file=None, evo_file=None, evo_flag=False)
L_dataset, _ = load_data_for_expert(os.path.join(data_path, dataset_name_L), dataset_name_L, mode="binary", gq_file=None, sq_file=None, evo_file=None, evo_flag=False)

# # C
# source_train_dataset = C_dataset["train"]
# source_val_dataset = C_dataset["val"]
# target_dataset = C_dataset["test"]
# # P
# source_train_dataset = P_dataset["train"]
# source_val_dataset = P_dataset["val"]
# target_dataset = P_dataset["test"]
#
# # G
# source_train_dataset = G_dataset["train"]
# source_val_dataset = G_dataset["val"]
# target_dataset = G_dataset["test"]

# # LIAR
source_train_dataset = L_dataset["train"]
source_val_dataset = L_dataset["val"]
target_dataset = L_dataset["test"]
# Create DataLoader instances
batch_size = 32  # Set your desired batch size

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'MESSAGE': sample['MESSAGE'],
            'label': sample['label']
        }

# Create custom datasets
source_train_dataset = CustomDataset(source_train_dataset)
source_val_dataset = CustomDataset(source_val_dataset )
target_dataset = CustomDataset(target_dataset )

# Create DataLoader instances
source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

# lode model

# model_name = 'roberta-base'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification
model.to("cuda")

# Training loop
epochs = 6  # You may need to adjust the number of epochs
best_val_loss = float('inf')

optimizer = AdamW(model.parameters(), lr=2e-5)  # You may need to adjust the learning rate
total_steps = len(source_train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    predictions_list = []
    labels_list = []
    best_val_accuracy = 0.0
    best_model_state_dict = None
    for batch in tqdm(source_train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        # Prepare input data
        inputs = tokenizer(batch['MESSAGE'], padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        labels_str = batch['label']
        labels = torch.tensor([1 if label.lower() == "true" else 0 for label in labels_str]).to(
            "cuda")  # Convert "true" to 1, "false" to 0
        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        # Save predictions and labels for later evaluation
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        predictions_list.extend(predictions.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        del inputs, labels
    avg_loss = total_loss / len(source_train_loader)
    accuracy = accuracy_score(labels_list, predictions_list)
    f1 = f1_score(labels_list, predictions_list)
    print(f'Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f} - F1 Score: {f1:.4f}')
    # Validation
    model.eval()
    val_loss = 0.
    val_predictions_list = []
    val_labels_list = []
    with torch.no_grad():
        for val_batch in source_val_loader:
            val_inputs = tokenizer(val_batch['MESSAGE'], padding=True, truncation=True, return_tensors="pt")
            val_inputs = {key: value.to("cuda") for key, value in val_inputs.items()}
            val_labels_str = val_batch['label']
            val_labels = torch.tensor([1 if label.lower() == "true" else 0 for label in val_labels_str]).to("cuda")  # Convert "true" to 1, "false" to 0

            val_outputs = model(**val_inputs, labels=val_labels)
            val_logits = val_outputs.logits
            val_predictions = torch.argmax(val_logits, dim=1)
            val_predictions_list.extend(val_predictions.cpu().numpy())
            val_labels_list.extend(val_labels.cpu().numpy())
            del val_inputs, val_labels
    accuracy_val = accuracy_score(val_labels_list, val_predictions_list)
    f1_val = f1_score(val_labels_list, val_predictions_list)

    # Save the best model based on validation accuracy
    if accuracy_val > best_val_accuracy:
        best_val_accuracy = accuracy_val
        best_model_state_dict = model.state_dict()

    print(f'Epoch {epoch + 1}/{epochs} - Val Accuracy: {accuracy_val:.4f} - Val F1 Score: {f1_val:.4f}')

# Load the best model for testing
del model
# best_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
best_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Assuming binary classification
best_model.load_state_dict(best_model_state_dict)
best_model.to("cuda")

# Testing on target dataset
best_model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch in target_loader:
        inputs = tokenizer(batch['MESSAGE'], padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}  # Move inputs to the same device as labels
        labels_str = batch['label']
        labels = torch.tensor([1 if label.lower() == "true" else 0 for label in labels_str]).to(
            "cuda")  # Convert "true" to 1, "false" to 0

        outputs = best_model(**inputs, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        test_predictions.extend(predictions.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        del inputs, labels

# Evaluate test results (you can use appropriate metrics)
accuracy_test = accuracy_score(test_labels, test_predictions)
f1_test = f1_score(test_labels, test_predictions)

print(
    f'Best Model - Test Accuracy: {accuracy_test:.4f} - Test F1 Score: {f1_test:.4f}')
