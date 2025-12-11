# Eliminate punctuation
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
# Download stopwords nltk if it is not already downloaded
nltk.download('stopwords')
# Read data from CSV
df = pd.read_csv('train_augmented.csv')

# Function to clear text
def clean_text(text):
 # Remove punctuation
 text = text.translate(str.maketrans('', '', string.punctuation))
 # Converts text to lowercase
 text = text.lower()
 # Removing stop words
 stop_words = set(stopwords.words('english'))
 text = ' '.join(word for word in text.split() if word not in
    stop_words and not word.isdigit())
 
 return text
# Apply the clean_text function to the review column
df['text'] = df['text'].apply(clean_text)
df.drop(columns=['id'], inplace=True)

# sentiment_mapping = {0: 'Very bad', 1: 'Bad', 2: 'Good', 3: 'Very good', 4: 'Excellent'}
# df['review'] = df['review'].map(sentiment_mapping)

# df = pd.DataFrame(df)
# pd.set_option('display.max_colwidth', None)
# print(df.head())
# pd.reset_option('display.max_colwidth')

from sklearn.model_selection import train_test_split
# Using 'Review' and 'Sentiment' columns for training
X = df['text']
y = df['review']

# Split data into training and testing data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from transformers import BertTokenizer
import torch
#Initialize Tokenizer and Encode Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the mapping from sentiment strings to integers
sentiment_to_int = {'Very bad': 0, 'Bad': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}

# Tokenization For BERT to understand
def encode_data(texts, labels, tokenizer, max_length=256):
    input_ids = []
    attention_masks = []
    # Convert string labels to numerical labels
    numerical_labels = [sentiment_to_int[label] for label in labels]

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(numerical_labels)

# Encode data
train_inputs, train_masks, train_labels = encode_data(X_train, y_train, tokenizer)
val_inputs, val_masks, val_labels = encode_data(X_val, y_val, tokenizer)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Shuffel the data and divide them into batches
batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


from transformers import BertForSequenceClassification
import torch
from torch.optim import AdamW

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False
)

# Definisikan optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)