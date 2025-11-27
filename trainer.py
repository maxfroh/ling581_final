import argparse
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value, DatasetDict
from transformers import AutoTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, EarlyStoppingCallback
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, confusion_matrix

age_ranges = [(13, 17), (23, 27), (33, 42)]


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def custom_compute_metrics(eval_prediction: EvalPrediction):
    logits, labels = eval_prediction
    preds = np.argmax(logits, axis=-1)

    # Basic metrics (macro + weighted)
    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    prec_macro = precision.compute(
        predictions=preds, references=labels, average="macro")["precision"]
    prec_weighted = precision.compute(
        predictions=preds, references=labels, average="weighted")["precision"]
    rec_macro = recall.compute(
        predictions=preds, references=labels, average="macro")["recall"]
    rec_weighted = recall.compute(
        predictions=preds, references=labels, average="weighted")["recall"]
    f1_macro = f1.compute(
        predictions=preds, references=labels, average="macro")["f1"]
    f1_weighted = f1.compute(
        predictions=preds, references=labels, average="weighted")["f1"]

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Put confusion matrix as a string (Trainer can't serialize numpy arrays)
    cm_str = np.array2string(cm)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm_str
    }


def create_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    return tokenizer


def create_dataset(tokenizer: AutoTokenizer, args: argparse.Namespace) -> DatasetDict:
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    desired_features = Features(
        {"label": Value("int64"), "text": Value("string")})
    dataset = Dataset.from_csv(
        "cleaned_data.csv", features=desired_features)

    # shrinking it down to just 10 samples for testing
    # dataset = dataset.train_test_split(test_size=1.5e-5)["test"]
    
    if args.shrink:
        # testing with the dataset being only 5% the original size
        dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)["test"]

    dataset = dataset.train_test_split(test_size=0.2, seed=args.seed)
    testing_split = dataset["test"].train_test_split(test_size=0.5, seed=args.seed)
    dataset["val"] = testing_split["train"]
    dataset["test"] = testing_split["test"]

    dataset = dataset.map(preprocess, batched=True)
    return dataset


def create_model(num_labels: int) -> RobertaForSequenceClassification:
    model = RobertaForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base", num_labels=num_labels, device_map="auto")
    return model


def create_trainer(tokenizer: AutoTokenizer, dataset: DatasetDict, model: RobertaForSequenceClassification, args: argparse.Namespace) -> Trainer:
    training_args = TrainingArguments(
        output_dir=f"./results_lr{args.learning_rate}_e{args.num_epochs}_b{args.batch_size}",
        logging_dir=f"./logs_lr{args.learning_rate}_e{args.num_epochs}_b{args.batch_size}",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
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
        compute_metrics=custom_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    return trainer


def main():
    parser = argparse.ArgumentParser(
        prog="RoBERTa Trainer",
        description="Builds a dataset and RoBERTa model, then trains it."
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5)
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    # parser.add_argument("--results_dir", type=str, default="./results")
    # parser.add_argument("--logs_dir", type=str, default="./logs")
    parser.add_argument("--shrink", action="store_true")
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    print(args)

    tokenizer = create_tokenizer()
    dataset = create_dataset(tokenizer, args)
    labels = set(dataset["train"]["label"])
    num_labels = len(labels)
    model = create_model(num_labels)
    trainer = create_trainer(tokenizer, dataset, model, args)
    
    print(dict(pd.Series(dataset["train"]["label"]).value_counts()))

    trainer.train()
    trainer.save_model(f"./saved_model_lr{args.learning_rate}_e{args.num_epochs}_b{args.batch_size}")
    trainer.evaluate(dataset["test"])


if __name__ == "__main__":
    main()
