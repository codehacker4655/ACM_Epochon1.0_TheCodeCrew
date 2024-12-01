from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the LIAR dataset
dataset = load_dataset("liar", trust_remote_code=True)

# Tokenization
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    # Adjust "statement" based on the actual dataset's structure
    return tokenizer(examples["statement"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Use smaller subsets of the dataset
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))  # 500 samples for training
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))  # 100 samples for evaluation

# Load pre-trained model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=6)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",           # Directory to save results
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=8,   # Smaller batch size
    gradient_accumulation_steps=2,   # Accumulate gradients to simulate larger batch size
    num_train_epochs=1,              # Only 1 epoch for quicker training
    fp16=True,                       # Enable mixed precision for faster training
    logging_dir="./logs",            # Directory to save logs
    logging_steps=10,                # Log every 10 steps
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("fake_news_model")
tokenizer.save_pretrained("fake_news_model")

