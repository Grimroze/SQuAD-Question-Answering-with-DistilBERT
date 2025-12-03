# !pip install torch torchvision torchaudio
# !pip install transformers datasets evaluate
# !pip install scikit-learn numpy pandas

import torch
import numpy as np
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
# from transformers import DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# SQuAD dataset (small version for quick training)
dataset = load_dataset("squad", split="train[:10%]")  # Only 10% to keep it manageable

print(f"Dataset size: {len(dataset)}")
print(f"\nFirst example:")
print(dataset[0])

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 384

def preprocess_function(examples):
    """
    Prepare data for BERT:
    1. Tokenize question and context
    2. Find where answers start and end in token space
    3. Create input_ids, attention_mask, token_type_ids
    """

    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # Tokenize question + context together
    encodings = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        padding="max_length",
        truncation="only_second",  # Truncate only context, not question
        return_offsets_mapping=True,
        return_overflowing_tokens=True  # Handle long contexts that get split into multiple inputs
    )

    start_positions = []
    end_positions = []

    # Iterate over each tokenized input (which might be a chunk of an original example)
    for i in range(len(encodings["input_ids"])):
        # Get the index of the original sample that this tokenized input (chunk) corresponds to
        sample_index = encodings["overflow_to_sample_mapping"][i]
        # Get the corresponding answer information for that original sample
        current_answers = examples["answers"][sample_index]

        offset_mapping = encodings["offset_mapping"][i]
        sequence_ids = encodings.sequence_ids(i)  # Identifies which tokens belong to question (0), context (1), or special tokens (None)

        # Default values for no answer (or answer not in this chunk)
        # A common practice is to point to the CLS token for unanswerable questions.
        cls_index = encodings["input_ids"][i].index(tokenizer.cls_token_id)
        start_token = cls_index
        end_token = cls_index

        # Only proceed if there's an actual answer for this sample
        if len(current_answers["text"]) > 0:
            answer_text = current_answers["text"][0]
            answer_start_char = current_answers["answer_start"][0]
            answer_end_char = answer_start_char + len(answer_text)

            # Find the start and end of the context in the current feature (chunk)
            context_start_token = None
            context_end_token = None
            for idx, seq_id in enumerate(sequence_ids):
                if seq_id == 1:  # This token belongs to the context
                    if context_start_token is None:
                        context_start_token = idx
                    context_end_token = idx

            # If context was found in this feature and the answer is within its character span
            if context_start_token is not None and context_end_token is not None:
                # Find the token range that covers the answer within the context
                for token_idx in range(context_start_token, context_end_token + 1):
                    start_char, end_char = offset_mapping[token_idx]

                    # If the answer starts within this token's range
                    if start_char <= answer_start_char < end_char:
                        start_token = token_idx
                    # If the answer ends within this token's range
                    if start_char < answer_end_char <= end_char:
                        end_token = token_idx

                # If the identified start/end tokens are not valid (e.g., answer split across chunks, or not in this chunk)
                # or if the answer goes beyond the bounds of the context in this specific chunk, revert to CLS token.
                if not (context_start_token <= start_token <= end_token <= context_end_token):
                    start_token = cls_index
                    end_token = cls_index
            else:
                # No context found in this feature (e.g., only question or special tokens)
                start_token = cls_index
                end_token = cls_index

        start_positions.append(start_token)
        end_positions.append(end_token)

    # Remove offset_mapping and overflow_to_sample_mapping as they are not needed for model input
    encodings.pop("offset_mapping")
    encodings.pop("overflow_to_sample_mapping", None)  # Use .pop(key, None) for safety as it might not always be present

    # Add the computed start and end positions to the encodings
    encodings["start_positions"] = start_positions
    encodings["end_positions"] = end_positions

    return encodings

# Apply preprocessing
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print(f"Processed dataset sample:")
print(processed_dataset[0])

# Splitting into train and validation
train_size = int(0.8 * len(processed_dataset))
val_size = len(processed_dataset) - train_size

train_dataset = processed_dataset.select(range(train_size))
val_dataset = processed_dataset.select(range(train_size, train_size + val_size))

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# Loading the model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Moving model to device (GPU or CPU)
model = model.to(device)

print(f"Model loaded: {model_name}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# Defining training arguments
training_args = TrainingArguments(
    output_dir="./qa_model",           # Where to save the model
    num_train_epochs=2,                # Train for 2 epochs (passes through data)
    per_device_train_batch_size=16,    # Process 16 examples at a time
    per_device_eval_batch_size=16,     # Validation batch size
    warmup_steps=500,                  # Gradually increase learning rate first
    weight_decay=0.01,                 # Regularization (prevents overfitting)
    logging_dir="./logs",              # Save training logs
    logging_steps=100,                 # Log every 100 steps
    eval_strategy="epoch",             # Evaluate after each epoch
    save_strategy="epoch",             # Save model after each epoch
    load_best_model_at_end=True,
    report_to="none"                   # Disable logging to Weights & Biases
)

# Creating a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Trainer initialized!")

trainer.train()

def answer_question(question, context):
    # Tokenize the input
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=384,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, dim=1)[0].item()
    end_idx = torch.argmax(outputs.end_logits, dim=1)[0].item()

    end_idx = end_idx + 1

    input_ids = inputs["input_ids"][0].tolist()
    answer_ids = input_ids[start_idx:end_idx]

    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return answer

print("Inference function ready!")

# Test with some examples
test_examples = [
    {
        "context": "Albert Einstein was born in Germany in 1879. He developed the theory of relativity.",
        "question": "Where was Albert Einstein born?"
    },
    {
        "context": "The Great Wall of China is one of the most impressive structures in the world. It was built over many centuries to protect against invasions.",
        "question": "What is the Great Wall of China?"
    },
    {
        "context": "Python is a popular programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        "question": "When was Python created?"
    }
]

print("=" * 80)
print("TESTING THE QUESTION ANSWERING MODEL")
print("=" * 80)

for i, example in enumerate(test_examples, 1):
    context = example["context"]
    question = example["question"]
    answer = answer_question(question, context)

    print(f"\n📝 Example {i}")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 80)
