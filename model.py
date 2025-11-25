import torch
import pandas as pd
from datasets import Dataset, Features, Value
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer

SEED = 7
age_ranges = [(13, 17), (23, 27), (33, 42)]


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

desired_features = Features({"label": Value("int64"), "text": Value("string")})
dataset = Dataset.from_csv("cleaned_data.csv", features=desired_features).train_test_split(test_size=1.5e-5)["test"]
dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
testing_split = dataset["test"].train_test_split(test_size=0.5, seed=SEED)
dataset["val"] = testing_split["train"]
dataset["test"] = testing_split["test"]

dataset = dataset.map(preprocess, batched=True)

labels = set(dataset["train"]["label"])
NUM_LABELS = len(labels)

print(dataset)

model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=NUM_LABELS)

LEARNING_RATE = 2e-5

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    processing_class=tokenizer,
)

trainer.train()


if __name__ == "__main__":
    pass