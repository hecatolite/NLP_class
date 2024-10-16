import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import random
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
import numpy as np

# 下载NLTK的punkt tokenizer
nltk.download('punkt')

# 数据预处理类
class TranslationDataset(Dataset):
    def __init__(self, lines, sp, vocab_en, vocab_jp, max_len=20):
        self.data = []
        self.max_len = max_len
        self.sp = sp
        self.vocab_en = vocab_en
        self.vocab_jp = vocab_jp
        # 处理传入的每一行
        for line in lines:
            jp_sent, en_sent = line.strip().split("\t")
            jp_tokenized = sp.encode_as_ids(jp_sent)[:max_len]
            en_tokenized = word_tokenize(en_sent.lower())[:max_len]
            en_tokenized = [vocab_en.get(w, vocab_en['<UNK>']) for w in en_tokenized]
            jp_tokenized = [vocab_jp.get(w, vocab_jp['<UNK>']) for w in jp_tokenized]
            self.data.append((jp_tokenized, en_tokenized))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        jp_sentence, en_sentence = self.data[idx]
        # 对句子进行填充
        if len(jp_sentence) < self.max_len:
            jp_sentence += [self.vocab_jp['<PAD>']] * (self.max_len - len(jp_sentence))
        if len(en_sentence) < self.max_len:
            en_sentence += [self.vocab_en['<PAD>']] * (self.max_len - len(en_sentence))
        return torch.tensor(jp_sentence), torch.tensor(en_sentence)


# 构建词汇表
def build_vocab(sentences, tokenizer, vocab_size=10000):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)
    vocab = {word: idx + 4 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
    
    # 添加特殊标记
    vocab['<PAD>'] = 0
    vocab['<SOS>'] = 1
    vocab['<EOS>'] = 2
    vocab['<UNK>'] = 3
    
    return vocab

# 使用sentencepiece处理日语
def tokenize_japanese(data_file):
    spm.SentencePieceTrainer.train('--input={} --model_prefix=spm_jp --vocab_size=10000'.format(data_file))
    sp = spm.SentencePieceProcessor(model_file='spm_jp.model')
    
    # 构建日语词汇表，并确保添加特殊标记
    vocab_jp = {sp.id_to_piece(id): id for id in range(10000)}
    vocab_jp['<PAD>'] = 0
    vocab_jp['<SOS>'] = 1
    vocab_jp['<EOS>'] = 2
    vocab_jp['<UNK>'] = 3
    
    # 返回sp和vocab_jp
    return sp, vocab_jp

# 加载和处理数据
def prepare_data(data_file, vocab_size=10000, max_len=20):
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读取所有行
    
    # 分离日语和英语句子
    jp_sentences = [line.split('\t')[0] for line in lines]
    en_sentences = [line.split('\t')[1] for line in lines]

    # 训练日语的sentencepiece模型并获取vocab_jp
    sp, vocab_jp = tokenize_japanese(data_file)

    # 构建英语词汇表
    vocab_en = build_vocab(en_sentences, word_tokenize, vocab_size=vocab_size)

    # 划分训练集、验证集、测试集
    data_size = len(lines)
    train_size = int(0.8 * data_size)
    val_size = int(0.1 * data_size)
    
    # 切分数据集，而不是传递文件路径
    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]

    train_data = TranslationDataset(train_lines, sp, vocab_en, vocab_jp, max_len=max_len)
    val_data = TranslationDataset(val_lines, sp, vocab_en, vocab_jp, max_len=max_len)
    test_data = TranslationDataset(test_lines, sp, vocab_en, vocab_jp, max_len=max_len)
    
    return train_data, val_data, test_data, vocab_en, vocab_jp, sp



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # encoder_outputs -> [batch_size, seq_len, hidden_dim]
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size_jp, vocab_size_en, embedding_dim, hidden_dim, output_dim, pad_idx, dropout=0.5):
        super(LSTMWithAttention, self).__init__()
        
        self.embedding_jp = nn.Embedding(vocab_size_jp, embedding_dim, padding_idx=pad_idx)
        self.embedding_en = nn.Embedding(vocab_size_en, embedding_dim, padding_idx=pad_idx)
        
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        
        self.attention = Attention(hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim * 2 + hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, jp_sentences, en_sentences):
        embedded_jp = self.dropout(self.embedding_jp(jp_sentences))
        encoder_outputs, (hidden, cell) = self.encoder(embedded_jp)
        
        embedded_en = self.dropout(self.embedding_en(en_sentences))
        
        # 用注意力机制处理每个时间步
        outputs = []
        for t in range(en_sentences.size(1)):
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            lstm_input = torch.cat((embedded_en[:, t], context), dim=1).unsqueeze(1)
            output, (hidden, cell) = self.decoder(lstm_input, (hidden, cell))
            output = self.fc_out(torch.cat((output.squeeze(1), context), dim=1))
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_bleu = 0
    for epoch in range(num_epochs):
        model.train()
        for jp_sentences, en_sentences in train_loader:
            optimizer.zero_grad()
            output = model(jp_sentences, en_sentences)
            loss = criterion(output.view(-1, output_dim), en_sentences.view(-1))
            loss.backward()
            optimizer.step()

        # 验证集评估
        bleu = evaluate_bleu(model, val_loader)
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), 'best_model.pth')
        print(f'Epoch {epoch+1}, BLEU: {bleu}')

def evaluate_bleu(model, data_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for jp_sentences, en_sentences in data_loader:
            output = model(jp_sentences, en_sentences)
            preds = torch.argmax(output, dim=2)
            predictions.extend(preds.tolist())
            targets.extend(en_sentences.tolist())

    # 计算 BLEU
    bleu_scores = []
    for pred, target in zip(predictions, targets):
        bleu = sentence_bleu([target], pred)
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)

# 加载数据
train_data, val_data, test_data, vocab_en, vocab_jp, sp = prepare_data('RNN_data/eng_jpn.txt')
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 初始化模型
embedding_dim = 256
hidden_dim = 512
output_dim = len(vocab_en)
pad_idx = vocab_jp['<PAD>']

model = LSTMWithAttention(len(vocab_jp), len(vocab_en), embedding_dim, hidden_dim, output_dim, pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# 测试集上评估
model.load_state_dict(torch.load('best_model.pth'))
test_bleu = evaluate_bleu(model, DataLoader(test_data, batch_size=32))
print(f'Test BLEU: {test_bleu}')
