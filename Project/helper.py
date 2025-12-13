from typing import Callable, Iterable, List, Optional, Tuple
import os

from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import tiktoken
import re
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer

# Comprehensive URL regex pattern
# Matches: http://, https://, www., IP addresses, and various domain formats
URL_PATTERN = re.compile(
    r'(?:https?://)?'  # Optional http:// or https://
    r'(?:www\.)?'      # Optional www.
    r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'  # Domain name
    r'[a-zA-Z]{2,}'    # Top-level domain
    r'(?::\d+)?'       # Optional port
    r'(?:/[^\s<>"{}|\\^`\[\]]*)?'  # Optional path
    r'(?:\?[^\s<>"{}|\\^`\[\]]*)?'  # Optional query string
    r'(?:#[^\s<>"{}|\\^`\[\]]*)?',  # Optional fragment
    re.IGNORECASE
)

def detect_urls(text):
    if pd.isna(text) or not isinstance(text, str):
        return False
    return bool(URL_PATTERN.search(text))

def extract_urls(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    return URL_PATTERN.findall(text)

def remove_urls(text, replacement=''):
    if pd.isna(text) or not isinstance(text, str):
        return text
    return URL_PATTERN.sub(replacement, text).strip()

def apply_url_processing(df, text_column='text', operation='remove', replacement=''):
    df = df.copy()
    
    if operation == 'detect':
        df['has_url'] = df[text_column].apply(detect_urls)
    elif operation == 'extract':
        df['urls'] = df[text_column].apply(extract_urls)
    elif operation == 'remove':
        df[text_column] = df[text_column].apply(
            lambda x: remove_urls(x, replacement)
        )
    else:
        raise ValueError("operation must be 'detect', 'extract', or 'remove'")
    
    return df


def target_distribution(df: pd.DataFrame, target_col: str = "review") -> pd.DataFrame:
    """
    Return counts and percentages for each target class.
    """
    counts = df[target_col].value_counts().sort_index()
    percents = df[target_col].value_counts(normalize=True).sort_index() * 100
    summary = pd.DataFrame({"count": counts, "percent": percents})
    return summary

def augment_with_bert_insert(
    texts: Iterable[str],
    model_path: str = "bert-base-uncased",
    n: int = 1,
    aug_p: float = 0.1,
    action: str = "insert",
    **kwargs,
) -> List[str]:

    aug = naw.ContextualWordEmbsAug(
        model_path=model_path,
        action=action,
        aug_p=aug_p,
        **kwargs,
    )
    results: List[str] = []
    for text in texts:
        augmented = aug.augment(text, n=n)
        # nlpaug returns a single string when n=1; normalize to list
        if isinstance(augmented, str):
            results.append(augmented)
        else:
            results.extend(augmented)
    return results

def augment_classes(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    classes: Iterable[str],
    augment_fn: Callable[[List[str]], List[str]],
    n_per_sample: int = 1,
    target_count: Optional[int] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)
    outputs = [df]

    counts = df[target_col].value_counts()
    max_count = counts.max()
    target = target_count or max_count

    for cls in classes:
        if cls not in counts:
            continue
        needed = target - counts[cls]
        if needed <= 0:
            continue

        cls_rows = df[df[target_col] == cls]
        source_n = math.ceil(needed / n_per_sample)
        sampled = cls_rows.sample(n=source_n, replace=True, random_state=random_state)

        augmented_texts = augment_fn(sampled[text_col].tolist())
        augmented_texts = augmented_texts[:needed]

        aug_df = pd.DataFrame({
            text_col: augmented_texts,
            target_col: cls,
        })
        outputs.append(aug_df)

    return pd.concat(outputs, ignore_index=True)


class CustomTokenizer:
    def __init__(self, vocab_size=8000, method='bpe'):
        self.vocab_size = vocab_size
        self.method = method
        self.tokenizer = None
        
    def train_from_texts(self, texts, output_dir='./tokenizers'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save texts to file
        corpus_file = os.path.join(output_dir, 'corpus.txt')
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(str(text) + '\n')
        
        if self.method == 'bpe':
            # HuggingFace BPE Tokenizer
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                min_frequency=2
            )
            
            self.tokenizer.train([corpus_file], trainer)
            self.tokenizer.save(os.path.join(output_dir, "bpe_tokenizer.json"))
            
        # elif self.method == 'sentencepiece':
        #     # SentencePiece (Google's method - works better for small datasets)
        #     import sentencepiece as spm
            
        #     spm.SentencePieceTrainer.train(
        #         input=corpus_file,
        #         model_prefix=os.path.join(output_dir, 'spm'),
        #         vocab_size=self.vocab_size,
        #         character_coverage=0.9995,
        #         model_type='bpe',  # or 'unigram'
        #         pad_id=0,
        #         unk_id=1,
        #         bos_id=2,
        #         eos_id=3,
        #         max_sentence_length=512,
        #         user_defined_symbols=['[CLS]', '[SEP]', '[MASK]']
        #     )
            
        #     # Load the trained model
        #     self.tokenizer = spm.SentencePieceProcessor()
        #     self.tokenizer.load(os.path.join(output_dir, 'spm.model'))
            
        # elif self.method == 'wordpiece':
        #     # WordPiece (BERT-style)
        #     self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        #     self.tokenizer.pre_tokenizer = Whitespace()
            
        #     trainer = WordPieceTrainer(
        #         vocab_size=self.vocab_size,
        #         special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        #         continuing_subword_prefix="##"
        #     )
            
        #     self.tokenizer.train([corpus_file], trainer)
        #     self.tokenizer.save(os.path.join(output_dir, "wordpiece_tokenizer.json"))
        # else:
        #     raise ValueError(f"Unknown method: {self.method}. Choose 'bpe', 'sentencepiece', or 'wordpiece'")
        
        # print(f"✅ Tokenizer trained with vocab size: {self.vocab_size}")
        # print(f"   Method: {self.method}")
        # print(f"   Saved to: {output_dir}")
        # return self.tokenizer
    
    def encode_batch(self, texts, max_length=128, return_tensors=None):
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings to encode
            max_length: Maximum sequence length (padding/truncation)
            return_tensors: 'pt' for PyTorch tensors, None for lists
            
        Returns:
            Encoded token IDs (padded/truncated to max_length)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train_from_texts() first.")
        
        if self.method == 'sentencepiece':
            # SentencePiece encoding
            encoded_ids = []
            for text in texts:
                ids = self.tokenizer.encode(str(text), out_type=int, add_bos=True, add_eos=True)
                # Pad/truncate
                if len(ids) > max_length:
                    ids = ids[:max_length]
                else:
                    ids = ids + [0] * (max_length - len(ids))
                encoded_ids.append(ids)
            
            if return_tensors == 'pt':
                return torch.tensor(encoded_ids)
            return encoded_ids
        else:
            # HuggingFace tokenizers (BPE or WordPiece)
            # Base tokenizers library doesn't support padding/truncation in encode_batch
            encoding = self.tokenizer.encode_batch([str(text) for text in texts])
            
            # Extract input_ids and handle padding/truncation manually
            ids = []
            for enc in encoding:
                token_ids = enc.ids
                # Truncate if too long
                if len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                # Pad if too short (0 is typically the padding token)
                elif len(token_ids) < max_length:
                    token_ids = token_ids + [0] * (max_length - len(token_ids))
                ids.append(token_ids)
            
            if return_tensors == 'pt':
                return torch.tensor(ids)
            return ids
    
    def encode(self, text, max_length=128):
        """Encode a single text string."""
        return self.encode_batch([text], max_length=max_length)[0]
    
    def decode(self, token_ids):
        """Decode token IDs back to text."""
        if self.method == 'sentencepiece':
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids)


# ============================================================================
# Neural Network Models
# ============================================================================

class BalancedBERT(nn.Module):
    """
    Optimized for balanced dataset of ~12K samples
    Larger capacity than previous models but carefully regularized
    """
    def __init__(self, vocab_size, num_classes=5, max_len=128,
                 hidden_size=256, num_layers=4, num_heads=8,
                 intermediate_size=512, dropout=0.2):
        super().__init__()
        
        # Embeddings (slightly larger for balanced data)
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers with residual connections
        self.encoder_layers = nn.ModuleList([
            BalancedTransformerLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Multi-head attention pooling (better than just CLS)
        self.attention_pool = MultiHeadAttentionPooling(hidden_size, num_heads=4)
        
        # Enhanced classifier with multiple residual blocks
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),  # Slightly less dropout in later layers
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
        print(f"BalancedBERT parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _init_weights(self):
        """Better initialization for balanced data"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.7)  # Lower gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, 
                                   device=input_ids.device).unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        words_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = (input_ids != 0)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask.float()) * -10000.0
        
        # Encoder layers with residual connections
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)
        
        # Attention pooling over all tokens (better representation)
        pooled_output = self.attention_pool(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class BalancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with pre-norm and better initialization"""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.2):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network with gated linear unit
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Learnable scaling factors (for better gradient flow)
        self.gamma1 = nn.Parameter(torch.ones(hidden_size))
        self.gamma2 = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x, attention_mask):
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=(attention_mask.squeeze(1).squeeze(1) == 0)
        )
        x = x + self.dropout1(attn_output) * self.gamma1
        
        # Pre-norm feed-forward
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output) * self.gamma2
        
        return x


class MultiHeadAttentionPooling(nn.Module):
    """Context-aware attention pooling"""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Linear projections
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, attention_mask):
        batch_size = hidden_states.size(0)
        
        # Expand learnable query
        query = self.query.expand(batch_size, -1, -1)
        
        # Project
        Q = self.q_linear(query)
        K = self.k_linear(hidden_states)
        V = self.v_linear(hidden_states)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.squeeze(1).squeeze(1).unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )
        
        # Output projection
        pooled = self.out_proj(context).squeeze(1)
        
        return pooled


# ============================================================================
# Small-data utilities (vocabulary + preprocessing)
# ============================================================================

def create_optimal_vocabulary(texts: Iterable[str], target_size: int = 8000) -> dict:
    """
    Build a small-data-friendly vocabulary combining frequent words and subword n-grams.
    Returns a mapping token -> id.
    """
    from collections import Counter

    word_counts = Counter()
    char_counts = Counter()

    for text in texts:
        text = str(text).lower()
        words = re.findall(r"\b\w[\w'\-]+\b", text)
        word_counts.update(words)
        for word in words:
            for i in range(len(word) - 2):
                char_counts[word[i : i + 3]] += 1

    vocab: List[str] = []
    # Top frequent words
    vocab.extend([w for w, _ in word_counts.most_common(6000)])

    # Mid-frequency informative words (<10% docs) using a simple doc freq heuristic
    total_docs = len(texts)
    for word, count in word_counts.items():
        if 5 <= count <= 20:
            doc_freq = sum(1 for t in texts if word in str(t))
            if doc_freq <= total_docs * 0.1:
                vocab.append(word)

    # Common character 3-grams as subwords
    vocab.extend([f"##{chars}##" for chars, _ in char_counts.most_common(1000)])

    # Limit and add specials
    vocab = list(dict.fromkeys(vocab))  # preserve order, dedupe
    vocab = vocab[: target_size - 3]
    vocab = ['<PAD>', '<UNK>', '<NUM>'] + vocab
    return {tok: idx for idx, tok in enumerate(vocab)}


def preprocess_text_for_small_data(text: str, vocab: dict, max_len: int = 150) -> List[int]:
    """
    Conservative preprocessing + token-to-id with subword fallback and dynamic padding/truncation.
    """
    text = str(text).lower()
    text = re.sub(r"\d+", " <NUM> ", text)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"([!?.])", r" \1 ", text)

    tokens: List[str] = []
    for part in text.split():
        if part in ['.', '!', '?', ',', ';', ':']:
            tokens.append(part)
        else:
            tokens.extend(re.findall(r"\b\w[\w'\-]+\b", part))

    encoded: List[int] = []
    for tok in tokens:
        if tok in vocab:
            encoded.append(vocab[tok])
        else:
            found = False
            for i in range(3, len(tok)):
                sub = f"##{tok[i-3:i]}##"
                if sub in vocab:
                    encoded.append(vocab[sub])
                    found = True
                    break
            if not found:
                encoded.append(vocab['<UNK>'])

    if len(encoded) > max_len:
        keep_start = encoded[: max_len // 2]
        keep_end = encoded[-(max_len // 2) :]
        encoded = keep_start + keep_end
    else:
        encoded = encoded + [vocab['<PAD>']] * (max_len - len(encoded))

    return encoded[:max_len]


# ============================================================================
# TextCNN with GLU for small datasets
# ============================================================================

class TextCNNGLU(nn.Module):
    """
    CNN with Gated Linear Units and attention pooling.
    Works well on smaller datasets.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)

        self.glu = nn.GLU(dim=1)

        self.attention = nn.Sequential(
            nn.Linear(96, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(96, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(48, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.embed_dropout(x)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        c3 = self.glu(self.conv3(x))
        c5 = self.glu(self.conv5(x))
        c7 = self.glu(self.conv7(x))

        combined = torch.cat([c3, c5, c7], dim=1).transpose(1, 2)  # (batch, seq_len, 96)
        attn_weights = F.softmax(self.attention(combined), dim=1)  # (batch, seq_len, 1)
        weighted = torch.sum(attn_weights * combined, dim=1)  # (batch, 96)

        logits = self.classifier(weighted)
        return logits, attn_weights