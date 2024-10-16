import MeCab
import nltk
import torch
from sklearn.model_selection import train_test_split

# 初始化Mecab
mecab_tagger = MeCab.Tagger("-Owakati")

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data from eng_jpn.txt
data_file = "RNN_data/eng_jpn.txt"
with open(data_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Split Japanese and English sentences
jp_sentences = [line.split('\t')[0] for line in lines]
en_sentences = [line.split('\t')[1].strip() for line in lines]

# 使用Mecab对日文句子进行
# 使用nltk对英文句子进行分词
nltk.download('punkt')
tokenized_jp = [mecab_tagger.parse(sentence).strip().split() for sentence in jp_sentences]
tokenized_en = [nltk.word_tokenize(sentence) for sentence in en_sentences]
#print(tokenized_jp[:5])
#print(tokenized_en[:5])

# 分割训练集，评估集和测试集（已经是tokenized）
train_jp, test_jp, train_en, test_en = train_test_split(tokenized_jp, tokenized_en, test_size=0.2, random_state=42)
val_jp, test_jp, val_en, test_en = train_test_split(test_jp, test_en, test_size=0.5, random_state=42)

# 从训练集构建日文和英文的词汇表
from collections import Counter

def build_vocab(tokenized_sentences, min_freq=1):
    word_freq = Counter()
    for sentence in tokenized_sentences:
        word_freq.update(sentence)
    vocab = {word: idx + 4 for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq}
    vocab['<PAD>'] = 0  # 填充符
    vocab['<SOS>'] = 1  # 句子起始符
    vocab['<EOS>'] = 2  # 句子结束符
    vocab['<UNK>'] = 3  # 未知词
    return vocab

# 构建词汇表
jp_vocab = build_vocab(train_jp)
en_vocab = build_vocab(train_en)

# 创建索引到词的映射
jp_idx2word = {idx: word for word, idx in jp_vocab.items()}
en_idx2word = {idx: word for word, idx in en_vocab.items()}

# 将句子转换为索引序列
def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in sentence]

from gensim.models import Word2Vec

# 如果没有训练好的模型，就训练一个
try:
    # 加载训练好的模型
    w2v_model = Word2Vec.load("word2vec_embeddings.model")
except FileNotFoundError:
    # 训练Word2Vec模型（只用训练日文的embedding）
    embedding_dim = 100
    w2v_model = Word2Vec(sentences=train_jp, vector_size=embedding_dim, window=5, min_count=1, sg=0)
    # 保存模型
    w2v_model.save("word2vec_embeddings.model")

# 测试W2V
word = "走れ"
try:
    similar_words = w2v_model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}':")
    for similar_word, score in similar_words:
        print(f"  {similar_word} ({score:.4f})")
except KeyError:
    print(f"The word '{word}' was not found in the vocabulary.")

# 获取日文的嵌入矩阵，词汇表和词汇到索引的映射
jp_vocab = w2v_model.wv.index_to_key
jp_word2idx = {word: idx for idx, word in enumerate(jp_vocab)}
jp_embedding_matrix = torch.tensor(w2v_model.wv.vectors)
# 加入PAD和UNK
jp_word2idx['<PAD>'] = len(jp_word2idx)
jp_word2idx['<UNK>'] = len(jp_word2idx)

# 构建英文词汇到索引的映射
en_word2idx = {word: idx for idx, word in enumerate(set(word for sentence in train_en for word in sentence))}
en_word2idx['<PAD>'] = len(en_word2idx)
en_word2idx['<UNK>'] = len(en_word2idx)

import torch.nn as nn
import torch.optim as optim

class LSTMTranslator(nn.Module):
    def __init__(self, input_dim, output_dim, jp_embedding_matrix, hidden_dim, num_layers=1):
        super(LSTMTranslator, self).__init__()
        
        # 日文嵌入层
        self.jp_embedding = nn.Embedding.from_pretrained(jp_embedding_matrix)
        
        # LSTM 层
        self.lstm = nn.LSTM(jp_embedding_matrix.shape[1], hidden_dim, num_layers, batch_first=True)
        
        # 全连接层，将 LSTM 输出转换为英文词汇表的概率分布
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        # 日文输入 -> 日文嵌入
        embedded_jp = self.jp_embedding(src)
        
        # LSTM 层处理嵌入
        outputs, _ = self.lstm(embedded_jp)
        
        # 输出通过全连接层转换为英文词的概率分布
        outputs = self.fc(outputs)
        
        return outputs

# 初始化模型
hidden_dim = 256
model = LSTMTranslator(len(jp_vocab), len(en_word2idx), jp_embedding_matrix, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# 用GPU训练模型
def train_model(model, train_jp, train_en, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for jp, en in zip(train_jp, train_en):
            jp = torch.tensor([jp_word2idx.get(word, jp_word2idx['<UNK>']) for word in jp]).unsqueeze(0).to(device)
            en = torch.tensor([en_word2idx.get(word, en_word2idx['<UNK>']) for word in en]).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            output = model(jp)
            loss = criterion(output.view(-1, len(en_word2idx)), en.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_jp)}")

try:
    # 加载预训练模型
    model.load_state_dict(torch.load("lstm.pth"))
except FileNotFoundError:
    # 训练模型
    train_model(model, train_jp, train_en)
    # 保存模型
    torch.save(model.state_dict(), "lstm.pth")

def translate_sentence(model, sentence):
    model.eval()
    with torch.no_grad():
        jp = torch.tensor([jp_word2idx.get(word, jp_word2idx['<UNK>']) for word in sentence]).unsqueeze(0).to(device)
        output = model(jp)
        output = output.squeeze(0)
        output = output.argmax(dim=-1)
        translation = [list(en_word2idx.keys())[idx] for idx in output]
        return translation

# 测试模型
test_sentence = "天気 が いい から 、 散歩 し ましょ う".split()
translation = translate_sentence(model, test_sentence)
print("Translated Sentence:", translation)