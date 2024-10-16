'''
1. tokenization
2. train word embeddings(CBOW) and eval
3. LSTM with pre-trained embeddings
4. Evaluate your LSTM with BLEU score and Perplexity
'''
import random
import sentencepiece as spm
from collections import Counter
from sklearn.model_selection import train_test_split

# Load data from eng_jpn.txt
data_file = "RNN_data/eng_jpn.txt"
with open(data_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Split Japanese and English sentences
jp_sentences = [line.split('\t')[0] for line in lines]
en_sentences = [line.split('\t')[1].strip() for line in lines]

# Split into train, validation, and test sets (8:1:1)
train_jp, test_jp, train_en, test_en = train_test_split(jp_sentences, en_sentences, test_size=0.2, random_state=42)
val_jp, test_jp, val_en, test_en = train_test_split(test_jp, test_en, test_size=0.5, random_state=42)


# Build Vocabulary Function
def build_vocab(sentences, vocab_size=10000):
    word_count = Counter()
    for sentence in sentences:
        word_count.update(sentence.split())
    most_common_words = word_count.most_common(vocab_size)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    return vocab

vocab_size = 10000
# Build vocabularies for both languages
vocab_jp = build_vocab(jp_sentences, vocab_size)
vocab_en = build_vocab(en_sentences, vocab_size)

# Example usage
print(f"Japanese Vocab Size: {len(vocab_jp)}")
print(f"English Vocab Size: {len(vocab_en)}")

from gensim.models import Word2Vec

# Combine all Japanese and English sentences for training embeddings
all_sentences = train_jp + train_en
tokenized_sentences = [sentence.split() for sentence in all_sentences]  # Tokenize

# Train Word2Vec model
embedding_dim = 100
model = Word2Vec(sentences=tokenized_sentences, vector_size=embedding_dim, window=5, min_count=1, sg=0)     # sg=0: CBOW, sg=1: Skip-gram

# eval
# Get the most similar words to a given word
word = "本"  # Japanese for "book"
try:
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}':")
    for similar_word, score in similar_words:
        print(f"  {similar_word} ({score:.4f})")
except KeyError:
    print(f"The word '{word}' was not found in the vocabulary.")

model.save("word2vec_embeddings.model")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, input_sentences, target_sentences, input_vocab, target_vocab, max_len=20):
        self.input_sentences = input_sentences
        self.target_sentences = target_sentences
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, idx):
        jp_tokens = self.input_sentences[idx].split()
        en_tokens = self.target_sentences[idx].split()

        # Convert tokens to indices and pad sequences
        jp_indices = [self.input_vocab.get(token, self.input_vocab['<UNK>']) for token in jp_tokens]
        en_indices = [self.target_vocab.get(token, self.target_vocab['<UNK>']) for token in en_tokens]

        # Padding
        jp_indices = jp_indices[:self.max_len] + [self.input_vocab['<PAD>']] * (self.max_len - len(jp_indices))
        en_indices = en_indices[:self.max_len] + [self.target_vocab['<PAD>']] * (self.max_len - len(en_indices))

        return torch.tensor(jp_indices), torch.tensor(en_indices)

# Define LSTM Model with Attention
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
        super(LSTMAttentionModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)

        embedded_trg = self.embedding(trg)
        decoder_outputs, _ = self.decoder(embedded_trg, (hidden, cell))

        output = self.fc(decoder_outputs)
        return output

# Hyperparameters
input_dim = len(vocab_jp)
output_dim = len(vocab_en)
embedding_dim = 100
hidden_dim = 128

# Model, Loss, Optimizer
model = LSTMAttentionModel(input_dim, output_dim, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_en['<PAD>'])
optimizer = optim.Adam(model.parameters())

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for src, trg in train_loader:
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train_dataset = TranslationDataset(train_jp, train_en, vocab_jp, vocab_en)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

train_model(model, train_loader, criterion, optimizer)

from nltk.translate.bleu_score import sentence_bleu

# BLEU score evaluation
def evaluate_bleu(model, data_loader):
    model.eval()
    total_bleu = 0
    for src, trg in data_loader:
        with torch.no_grad():
            output = model(src, trg[:, :-1])
            predicted = output.argmax(2)
            for i in range(predicted.shape[0]):
                reference = trg[i, 1:].tolist()
                hypothesis = predicted[i].tolist()
                total_bleu += sentence_bleu([reference], hypothesis)
    return total_bleu / len(data_loader)

# Calculate BLEU and Perplexity
bleu_score = evaluate_bleu(model, train_loader)
print(f'BLEU Score: {bleu_score}')

# Perplexity calculation
def calculate_perplexity(loss):
    return torch.exp(loss)

perplexity = calculate_perplexity(loss)
print(f'Perplexity: {perplexity.item()}')
