import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Constants
EMBEDDING_DIM = 100
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
DROPOUT_RATE = 0.5
L2_CONSTRAINT = 3
BATCH_SIZE = 50
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Step 1: Data Preprocessing
class TextDataset(Dataset):
    def __init__(self, data_file, vocab=None, max_len=100):
        self.sentences = []
        self.labels = []
        self.vocab = vocab
        self.max_len = max_len
        self.build_dataset(data_file)
    
    def build_dataset(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            sentence, label = line.strip().split("\t")
            words = list(jieba.cut(sentence))
            if self.vocab is not None:
                words = [self.vocab[word] if word in self.vocab else self.vocab['<UNK>'] for word in words]
            self.sentences.append(words)
            self.labels.append(int(label))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        # Pad sequence
        if len(sentence) < self.max_len:
            sentence = sentence + [self.vocab['<PAD>']] * (self.max_len - len(sentence))
        else:
            sentence = sentence[:self.max_len]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 把train_file中的句子分词，然后构建词表
def build_vocab(data_file):
    vocab = defaultdict(lambda: len(vocab))
    vocab['<PAD>']  # padding token
    vocab['<UNK>']  # unknown token
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sentence, _ = line.strip().split("\t")
        words = list(jieba.cut(sentence))
        for word in words:
            vocab[word]
    return vocab

# Step 2: CNN Model for Sentence Classification
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, dropout_rate, num_classes):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, embedding_dim)) for filter_size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch_size, 1, max_len, embedding_dim]
        conv_results = [torch.relu(conv(x)).squeeze(3) for conv in self.conv_layers]  # Apply Conv layers
        pooled = [torch.max(conv_result, 2)[0] for conv_result in conv_results]  # Max-over-time pooling
        concat = torch.cat(pooled, 1)  # Concatenate along the feature dimension
        dropout_output = self.dropout(concat)
        return self.fc(dropout_output)

# Step 3: Training and Evaluation

def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=20, patience=5):
    best_dev_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        dev_loss, dev_acc = evaluate_model(model, dev_loader, criterion)
        
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}')
        
        # Early Stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print('Early stopping triggered.')
            break

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# Step 4: Load Data and Prepare for Training
if __name__ == "__main__":
    train_file = 'CNN_data/train.txt'
    dev_file = 'CNN_data/dev.txt'
    test_file = 'CNN_data/test.txt'
    
    # Build vocabulary
    vocab = build_vocab(train_file)
    
    # Prepare datasets
    # 把句子转化成词索引的序列
    train_dataset = TextDataset(train_file, vocab)
    dev_dataset = TextDataset(dev_file, vocab)
    test_dataset = TextDataset(test_file, vocab)
    
    # 这一步是为了把词索引的序列转化成tensor，是训练常用的操作
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    vocab_size = len(vocab)
    num_classes = len(set([label for _, label in train_dataset]))
    model = CNNTextClassifier(vocab_size, EMBEDDING_DIM, FILTER_SIZES, NUM_FILTERS, DROPOUT_RATE, num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    
    # Train model with early stopping
    train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)
    
    # Evaluate on test set
    model.load_state_dict(torch.load('best_model.pt')) # 加载最好的模型
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
