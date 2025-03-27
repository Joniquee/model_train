
class CustomTrainer:
    def __init__(self, model, train_dataset, eval_dataset, eval_steps, tokenizer, device, batch_size_train=64,
                 batch_size_eval=128, lr=5e-5, epochs=3, log_dir='./logs'):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.lr = lr
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.log_dir = log_dir

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                           collate_fn=default_data_collator)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size_eval, collate_fn=default_data_collator)

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = None

        self.global_step = 0

    def train(self):
        num_training_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        print("IM HERE")

        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            total_correct, total_samples = 0, 0
            all_preds = []
            all_labels = []
            loop = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}",
                        total=len(self.train_dataloader))
            for batch_idx, batch in enumerate(loop):
                print(f"Batch {batch_idx + 1}/{len(self.train_dataloader)}")
                batch = {k: v.to(self.device) for k, v in batch.items() if
                         k in ['input_ids', 'attention_mask', 'labels']}

                self.optimizer.zero_grad()

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                total_correct += (preds == labels).sum()
                total_samples += labels.size

                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                accuracy = total_correct / total_samples

                if (batch_idx + 1) % 10 == 0:
                    precision = precision_score(all_labels, all_preds, average='macro')
                    recall = recall_score(all_labels, all_preds, average='macro')
                    f1 = f1_score(all_labels, all_preds, average='macro')
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(
                        f"Train Batch {batch_idx + 1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

                loop.set_postfix(
                    loss=loss.item(), accuracy=accuracy, lr=self.scheduler.get_last_lr()[0]
                )
                if self.global_step % self.eval_steps == 0 and self.global_step != 0:
                    print("mb here")
                    self.evaluate()

                self.global_step += 1

            print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss / len(self.train_dataloader)}")

            self.save_model()

    def evaluate(self):
        self.model.eval()
        eval_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            loop = tqdm(self.eval_dataloader, desc="Evaluating", total=len(self.eval_dataloader))
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items() if
                         k in ['input_ids', 'attention_mask', 'labels']}

                outputs = self.model(**batch)
                eval_loss += outputs.loss.item()

                preds = outputs.logits.argmax(dim=-1)
                labels = batch['labels']
                correct_predictions += (preds == labels).sum().item()
                total_predictions += len(labels)

                loop.set_postfix(eval_loss=eval_loss / (loop.n + 1), accuracy=correct_predictions / total_predictions)

        avg_loss = eval_loss / len(self.eval_dataloader)
        accuracy = correct_predictions / total_predictions

        print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")
        self.model.train()

    def save_model(self, path='kaggle/working/model'):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def log_metrics(self, epoch, loss, accuracy):
        pass


trainer = CustomTrainer(
    model=model,
    train_dataset=tokenized_dataset['train'], eval_dataset=tokenized_dataset['test'],
    eval_steps=len(tokenized_dataset['train']),
    tokenizer=tokenizer,
    device=device,
    batch_size_train=32,
    batch_size_eval=64,
    lr=5e-5,
    epochs=3,
)

try:
    trainer.train()
except KeyboardInterrupt:
    print("Сохраняем модель")
    model.save_pretrained("/kaggle/working/model")
    tokenizer.save_pretrained("/kaggle/working/model/tokenizer")
    print("Готово")

if __name__ == "__main__":
    main()