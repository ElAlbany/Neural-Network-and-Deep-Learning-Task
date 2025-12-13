# Complete Code Explanation

## helper.py - Line by Line Explanation

### Imports (Lines 1-17)

**Lines 1-2**: Type hints and OS module
- `from typing import ...`: Type hints for function signatures
- `import os`: File system operations

**Lines 4-7**: Tokenizer libraries
- `from tokenizers import ...`: HuggingFace tokenizers library for BPE/WordPiece
- `from tokenizers.models import ...`: Tokenization models (BPE, WordPiece)
- `from tokenizers.trainers import ...`: Trainers to build vocabularies
- `from tokenizers.pre_tokenizers import ...`: Preprocessing (whitespace splitting)

**Line 8**: `import tiktoken`: OpenAI's tokenizer (not used in active code)

**Lines 9-12**: Core libraries
- `import re`: Regular expressions for text processing
- `import math`: Mathematical operations
- `import numpy as np`: Numerical computing
- `import pandas as pd`: Data manipulation

**Lines 13-15**: PyTorch
- `import torch`: PyTorch tensor operations
- `import torch.nn as nn`: Neural network modules
- `import torch.nn.functional as F`: Functional operations (activation functions, etc.)

**Lines 16-17**: Data augmentation
- `import nlpaug.augmenter.word as naw`: NLP augmentation library
- `from transformers import ...`: HuggingFace transformers (not actively used)

### URL Processing (Lines 19-62)

**Lines 19-31**: URL regex pattern
- `URL_PATTERN = re.compile(...)`: Compiled regex to match URLs
  - `(?:https?://)?`: Optional http:// or https://
  - `(?:www\.)?`: Optional www.
  - Domain pattern: Matches valid domain names
  - `[a-zA-Z]{2,}`: Top-level domain (at least 2 letters)
  - `(?::\d+)?`: Optional port number
  - Path, query string, and fragment patterns
  - `re.IGNORECASE`: Case-insensitive matching

**Lines 33-36**: `detect_urls()` function
- Checks if text is valid string
- Returns True if URL pattern found, False otherwise

**Lines 38-41**: `extract_urls()` function
- Returns list of all URLs found in text
- Returns empty list if no URLs or invalid input

**Lines 43-46**: `remove_urls()` function
- Replaces URLs with replacement string (default empty)
- Strips whitespace from result

**Lines 48-62**: `apply_url_processing()` function
- Applies URL operations to entire DataFrame
- `df = df.copy()`: Creates copy to avoid modifying original
- Three operations:
  - `'detect'`: Adds `has_url` boolean column
  - `'extract'`: Adds `urls` column with list of URLs
  - `'remove'`: Removes URLs from text column

### Target Distribution (Lines 65-72)

**Lines 65-72**: `target_distribution()` function
- `counts = ...`: Counts occurrences of each class
- `percents = ...`: Calculates percentages (normalized counts × 100)
- Returns DataFrame with count and percent columns

### Data Augmentation (Lines 74-158)

**Lines 74-106**: `augment_with_bert_insert()` function
- Uses BERT to insert/replace words contextually
- `aug = naw.ContextualWordEmbsAug(...)`: Creates BERT-based augmenter
  - `model_path`: BERT model to use
  - `action`: "insert" or "substitute"
  - `aug_p`: Proportion of tokens to augment
- Loops through texts, augments each, normalizes output format

**Lines 108-158**: `augment_classes()` function
- Augments minority classes to balance dataset
- `rng = np.random.default_rng(random_state)`: Random number generator
- `outputs = [df]`: Start with original dataframe
- `counts = df[target_col].value_counts()`: Count samples per class
- `max_count = counts.max()`: Largest class size
- `target = target_count or max_count`: Target size (defaults to max)
- For each class to augment:
  - Calculates how many samples needed
  - Samples with replacement from existing class samples
  - Applies augmentation function
  - Creates new DataFrame with augmented texts
  - Appends to outputs
- Returns concatenated DataFrame with original + augmented data

### Custom Tokenizer (Lines 161-313)

**Lines 161-174**: `CustomTokenizer` class initialization
- `vocab_size`: Size of vocabulary to build
- `method`: Tokenization method ('bpe', 'sentencepiece', 'wordpiece')
- `self.tokenizer = None`: Will hold trained tokenizer

**Lines 176-250**: `train_from_texts()` method
- `os.makedirs(output_dir, exist_ok=True)`: Creates output directory
- Saves all texts to `corpus.txt` file (one per line)
- If method is 'bpe':
  - Creates HuggingFace BPE tokenizer
  - Sets whitespace pre-tokenizer
  - Creates BpeTrainer with vocab size and special tokens
  - Trains on corpus file
  - Saves to JSON file
- (SentencePiece and WordPiece code is commented out)

**Lines 252-301**: `encode_batch()` method
- Encodes multiple texts at once
- If SentencePiece: encodes each text, pads/truncates to max_length
- Otherwise (BPE/WordPiece):
  - Uses HuggingFace tokenizer's encode_batch
  - Manually handles padding/truncation
  - Returns PyTorch tensor if `return_tensors='pt'`, else list

**Lines 303-305**: `encode()` method
- Wrapper to encode single text (calls encode_batch with one item)

**Lines 307-312**: `decode()` method
- Decodes token IDs back to text string

### Neural Network Models (Lines 315-510)

**Lines 319-405**: `BalancedBERT` class
- Transformer-based model for text classification

**Lines 324-327**: `__init__()` parameters
- `vocab_size`: Size of token vocabulary
- `num_classes`: Number of output classes (5 for sentiment)
- `max_len`: Maximum sequence length (128)
- `hidden_size`: Embedding dimension (256)
- `num_layers`: Number of transformer layers (4)
- `num_heads`: Attention heads per layer (8)
- `intermediate_size`: FFN hidden size (512)
- `dropout`: Dropout rate (0.2)

**Lines 329-332**: Embedding layers
- `self.embeddings`: Word embeddings (vocab_size → hidden_size)
- `self.position_embeddings`: Position embeddings (max_len → hidden_size)
- `self.token_type_embeddings`: Token type embeddings (2 → hidden_size, for BERT-style)

**Lines 334-335**: Normalization and dropout
- `LayerNorm`: Normalizes embeddings
- `dropout`: Regularization layer

**Lines 338-341**: Transformer encoder
- Creates list of `BalancedTransformerLayer` modules
- 4 layers stacked

**Line 344**: Attention pooling
- `MultiHeadAttentionPooling`: Pools all tokens into single vector

**Lines 347-355**: Classifier head
- Three-layer MLP:
  - hidden_size → hidden_size (with GELU, dropout)
  - hidden_size → hidden_size//2 (with GELU, less dropout)
  - hidden_size//2 → num_classes (output logits)

**Line 357**: Weight initialization
- Calls `_init_weights()` method

**Line 358**: Prints total parameter count

**Lines 360-368**: `_init_weights()` method
- Initializes all weights:
  - Linear layers: Xavier uniform with gain 0.7
  - Embeddings: Normal distribution (mean=0, std=0.02)

**Lines 370-405**: `forward()` method
- **Line 371**: Gets sequence length from input
- **Lines 372-373**: Creates position IDs (0 to seq_length-1)
- **Lines 375-376**: Creates token type IDs (all zeros if not provided)
- **Lines 379-381**: Computes three embedding types
- **Line 383**: Sums all embeddings
- **Lines 384-385**: Applies layer norm and dropout
- **Lines 388-392**: Creates attention mask (1 for real tokens, 0 for padding)
- **Lines 395-397**: Passes through transformer layers
- **Line 400**: Applies attention pooling
- **Line 403**: Gets classification logits
- **Line 405**: Returns logits

**Lines 408-452**: `BalancedTransformerLayer` class
- Single transformer layer with pre-norm architecture

**Lines 410-436**: `__init__()`
- `self.self_attn`: Multi-head self-attention
- `self.ffn`: Feed-forward network with GLU (Gated Linear Unit)
- `self.norm1`, `self.norm2`: Layer normalization before attention and FFN
- `self.dropout1`, `self.dropout2`: Dropout layers
- `self.gamma1`, `self.gamma2`: Learnable scaling factors for residual connections

**Lines 438-452**: `forward()` method
- Pre-norm attention: normalizes input, applies attention, adds residual
- Pre-norm FFN: normalizes, applies FFN, adds residual
- Uses learnable gammas to scale residual connections

**Lines 455-510**: `MultiHeadAttentionPooling` class
- Attention-based pooling mechanism

**Lines 457-469**: `__init__()`
- `num_heads`: Number of attention heads
- `head_dim`: Dimension per head (hidden_size // num_heads)
- `self.query`: Learnable query vector (1×1×hidden_size)
- `q_linear`, `k_linear`, `v_linear`: Linear projections for Q, K, V
- `out_proj`: Output projection

**Lines 471-510**: `forward()` method
- Expands learnable query to batch size
- Projects hidden states to Q, K, V
- Reshapes for multi-head attention
- Computes attention scores (scaled dot-product)
- Applies attention mask (sets padding to -inf)
- Applies softmax to get attention weights
- Computes weighted sum of values
- Reshapes and projects to get pooled output

### Small-Data Utilities (Lines 513-595)

**Lines 517-554**: `create_optimal_vocabulary()` function
- Builds vocabulary optimized for small datasets
- Counts words and character 3-grams
- Adds top 6000 frequent words
- Adds mid-frequency informative words (appear in <10% of docs)
- Adds common character 3-grams as subwords
- Limits to target_size, adds special tokens: `<PAD>`, `<UNK>`, `<NUM>`

**Lines 557-595**: `preprocess_text_for_small_data()` function
- Preprocesses text and converts to token IDs
- Lowercases text
- Replaces numbers with `<NUM>` token
- Normalizes punctuation
- Tokenizes words
- Maps tokens to IDs (with subword fallback)
- Truncates or pads to max_len

### TextCNN Model (Lines 598-655)

**Lines 602-655**: `TextCNNGLU` class
- CNN-based model with Gated Linear Units

**Lines 608-633**: `__init__()`
- `embedding`: Word embeddings
- `embed_dropout`: Dropout on embeddings
- `conv3`, `conv5`, `conv7`: 1D convolutions with kernel sizes 3, 5, 7
- `glu`: Gated Linear Unit activation
- `attention`: Attention mechanism for pooling
- `classifier`: Final classification layers

**Lines 635-639**: `_initialize_weights()`
- Xavier uniform for embeddings
- Kaiming normal for convolutions

**Lines 641-655**: `forward()` method
- Embeds input tokens
- Applies dropout
- Transposes for convolution (batch, embed_dim, seq_len)
- Applies three convolutions with GLU
- Concatenates outputs
- Applies attention pooling
- Classifies to get logits
- Returns logits and attention weights

---

## main.ipynb - Cell by Cell Explanation

### Cell 0: Imports
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helper
import importlib
importlib.reload(helper)
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
import sentencepiece as spm
import os
```
- Imports data science libraries (pandas, numpy, matplotlib)
- Imports helper module and reloads it (useful during development)
- Imports tokenizer libraries
- Imports OS for file operations

### Cell 1: Load Data
```python
df_original = pd.read_csv("train.csv")
df = df_original
```
- Loads training data from CSV
- Creates working copy `df`

### Cell 2: Check Unique Reviews
```python
df['review'].unique()
```
- Shows unique values in 'review' column (the 5 sentiment classes)

### Cell 3: Markdown - "Removing URLs"
- Documentation cell

### Cell 4: Detect URLs
```python
df_with_url_detection = helper.apply_url_processing(df, text_column='text', operation='detect')
print(f"Number of reviews with URLs: {df_with_url_detection['has_url'].sum()}")
print(f"Percentage: {df_with_url_detection['has_url'].mean() * 100:.2f}%")
```
- Detects URLs in text column
- Prints count and percentage of reviews with URLs

### Cell 5: Remove URLs
```python
pd.set_option('display.max_colwidth', None)
df_cleaned = helper.apply_url_processing(df, text_column='text', operation='remove', replacement='')
print(f"\n✅ Created 'text_no_urls' column with URLs removed")
```
- Sets pandas to show full column width
- Removes URLs from text
- Prints confirmation message

### Cell 6: Update DataFrame
```python
df = df_cleaned
```
- Updates working dataframe with cleaned version

### Cell 7: Markdown - "Converting to LowerCase"
- Documentation cell

### Cell 8: Lowercase Text
```python
df['text'] = df['text'].str.lower()
```
- Converts all text to lowercase for consistency

### Cell 9: Sample Text Check
```python
x = df.loc[df['id'] == 7961,'text']
print(x)
```
- Retrieves and prints a specific review to verify preprocessing

### Cell 10: Markdown - "Target Distribution"
- Documentation cell

### Cell 11: Analyze Target Distribution
```python
summary = helper.target_distribution(df, target_col="review")
print("Target distribution (count, percent):")
print(summary)
print(f"\nTotal samples: {len(df):,}")
print(f"Imbalance ratio (max/min): {summary['count'].max() / summary['count'].min():.2f}")
```
- Calculates class distribution
- Prints counts, percentages, total samples, and imbalance ratio

### Cell 12: Markdown - "Data Augmentation"
- Documentation cell

### Cell 13: Data Augmentation (Commented Out)
- Contains commented code for BERT-based augmentation
- When uncommented, augments minority classes to balance dataset

### Cell 14: Markdown - "Save Augmented Data"
- Documentation cell

### Cell 15: Save Augmented Data
```python
save_augmented = True
output_filename = 'train_augmented.csv'
if save_augmented:
    # Checks for augmented data and saves to CSV
    # Prints file info and preview
```
- Saves augmented dataset to CSV file
- Shows file size and preview

### Cell 16: Markdown - "Custom Tokenizer Training"
- Documentation cell

### Cell 17: Train Tokenizer
```python
train_tokenizer = True
tokenizer_method = 'bpe'
vocab_size = 8000
output_dir = './tokenizers'
if train_tokenizer:
    texts = df['text'].dropna().tolist()
    tokenizer = helper.CustomTokenizer(vocab_size=vocab_size, method=tokenizer_method)
    tokenizer.train_from_texts(texts, output_dir=output_dir)
    # Tests encoding on sample texts
```
- Trains BPE tokenizer on corpus
- Tests encoding on sample texts
- Saves tokenizer to file

### Cell 18: Markdown - "Model Training Setup"
- Documentation cell

### Cell 19: Prepare Training Data
```python
# Load augmented data if available
if os.path.exists('train_augmented.csv'):
    df_train = pd.read_csv('train_augmented.csv')
else:
    df_train = df.copy()

# Load or create tokenizer
if 'tokenizer' not in locals():
    # Loads saved tokenizer if exists

# Encode all texts
texts = df_train['text'].dropna().tolist()
encoded_texts = tokenizer.encode_batch(texts, max_length=128, return_tensors='pt')

# Create label mapping
label_map = {'Very bad': 0, 'Bad': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
labels = torch.tensor([label_map[label] for label in df_train['review'].values], dtype=torch.long)
```
- Loads augmented data or uses original
- Loads tokenizer (trains if needed)
- Encodes all texts to token IDs
- Converts labels to numeric IDs
- Prints data statistics

### Cell 20: Create Data Splits
```python
train_size = 0.7
val_size = 0.15
test_size = 0.15

# First split: train vs (val + test)
X_temp, X_test, y_temp, y_test = train_test_split(
    encoded_texts, labels, test_size=test_size, random_state=42, stratify=labels
)

# Second split: train vs val
val_size_adjusted = val_size / (train_size + val_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
)
```
- Splits data into 70% train, 15% validation, 15% test
- Uses stratified splitting to maintain class distribution
- Two-step split to get three sets

### Cell 21: Create DataLoaders
```python
class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'input_ids': self.texts[idx], 'labels': self.labels[idx]}

train_dataset = ReviewDataset(X_train, y_train)
val_dataset = ReviewDataset(X_val, y_val)
test_dataset = ReviewDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```
- Defines PyTorch Dataset class
- Creates datasets for train/val/test
- Creates DataLoaders with batch size 32
- Shuffles training data only

### Cell 22: Markdown - "Initialize Model"
- Documentation cell

### Cell 23: Initialize Model
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get vocab size from tokenizer
# (Multiple checks to determine vocab size)

# Model hyperparameters
num_classes = 5
max_len = 128
hidden_size = 256
num_layers = 4
num_heads = 8
intermediate_size = 512
dropout = 0.2

model = helper.BalancedBERT(
    vocab_size=vocab_size,
    num_classes=num_classes,
    max_len=max_len,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_heads=num_heads,
    intermediate_size=intermediate_size,
    dropout=dropout
).to(device)
```
- Detects available device (GPU or CPU)
- Determines vocabulary size from tokenizer
- Sets model hyperparameters
- Creates BalancedBERT model and moves to device

### Cell 24: Markdown - "Training Loop"
- Documentation cell

### Cell 25: Training Loop
```python
train_model = True
num_epochs = 10
learning_rate = 2e-4
weight_decay = 1e-4

if train_model:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Evaluate on validation set
        
        # Save best model based on validation accuracy
```
- Sets training hyperparameters
- Creates loss function (CrossEntropyLoss)
- Creates optimizer (AdamW with weight decay)
- Creates learning rate scheduler
- Training loop:
  - Sets model to training mode
  - Iterates through batches
  - Forward pass, backward pass, optimizer step
  - Calculates training accuracy
- Validation loop:
  - Sets model to evaluation mode
  - No gradients needed
  - Calculates validation loss and accuracy
- Saves best model based on validation accuracy

### Cell 26: Evaluate on Test Set (Has Error)
```python
if train_model and os.path.exists('best_model.pt'):
    # Loads best model
    # Evaluates on test set
    # Prints metrics
```
- This cell has an error: `train_model` variable not in scope
- Should evaluate best model on test set
- Would print classification report and confusion matrix

### Cell 27: Markdown - "Evaluate Saved Model on Test Set"
- Documentation cell

### Cell 28: Evaluate Saved Model
```python
model_path = 'best_model.pt'

if 'test_loader' not in locals():
    print("⚠️ test_loader not found...")
elif not os.path.exists(model_path):
    print(f"⚠️ {model_path} not found...")
else:
    # Recreate model if needed
    # Load weights
    # Evaluate on test set
    # Print results
```
- Standalone evaluation cell
- Checks for test_loader and model file
- Recreates model if needed
- Loads saved weights
- Evaluates on test set
- Prints detailed metrics

### Cell 29: Empty
- Empty cell (placeholder)

---

## Summary

This codebase implements a complete text classification pipeline:

1. **Data Preprocessing**: URL removal, lowercasing
2. **Data Augmentation**: BERT-based augmentation to balance classes
3. **Tokenization**: Custom BPE tokenizer trained on corpus
4. **Model**: BalancedBERT - a transformer-based classifier
5. **Training**: Full training loop with validation
6. **Evaluation**: Test set evaluation with detailed metrics

The model architecture uses:
- Transformer encoder with 4 layers
- Multi-head attention pooling
- Gated Linear Units (GLU) in feed-forward networks
- Pre-norm architecture for better gradient flow
- Careful regularization for small dataset

