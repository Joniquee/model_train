import datasets
import shutil
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from datasets import Dataset
import evaluate
import os
import csv
from decimal import Decimal

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        return metric.compute(predictions=predictions, references=labels)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2, problem_type="single_label_classification")
    model.to(device)

    #shutil.copytree("/kaggle/input/dataset3/tokenized_dataset2", "/kaggle/working/tokenized_dataset2")

    tokenized_dataset = datasets.load_from_disk("/kaggle/working/tokenized_dataset2")
    tokenized_dataset = tokenized_dataset.cast_column("labels", datasets.Value("int64"))


    training_args = TrainingArguments(
        output_dir='/kaggle/working/results',
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='/kaggle/working/logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy='epoch',
        load_best_model_at_end=True,
        fp16=True,
        run_name="distilbert_training_v1",
        dataloader_num_workers=4,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    try:
      trainer.train()
    except KeyboardInterrupt:
      print("Обучение прервано пользователем. Сохраняем модель...")
      output_dir = "/kaggle/working/trained_model"
      trainer.model.save_pretrained(output_dir)
      tokenizer.save_pretrained(output_dir)
      return

    output_dir = "/kaggle/working/trained_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()