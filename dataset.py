from transformers import DistilBertTokenizer
import pandas as pd
from datasets import Dataset



tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

df = pd.read_csv("/content/drive/MyDrive/untokenized_dataset/AI_Human.csv")

dataset = Dataset.from_list(df.to_dict(orient='records'))


def tokenize(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized["labels"] = examples["generated"]
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, train_size=0.8)
tokenized_dataset.save_to_disk("/content/drive/MyDrive/tokenized_dataset2")