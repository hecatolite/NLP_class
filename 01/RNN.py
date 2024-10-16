import MeCab
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

# 初始化 MeCab
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

# 使用 MeCab 对日文句子进行分词
# 使用 nltk 对英文句子进行分词
nltk.download('punkt')
tokenized_jp = [mecab_tagger.parse(sentence).strip().split() for sentence in jp_sentences]
tokenized_en = [nltk.word_tokenize(sentence.lower()) for sentence in en_sentences]

# 分割训练集、验证集和测试集（已经是 tokenized）
train_jp, test_jp, train_en, test_en = train_test_split(tokenized_jp, tokenized_en, test_size=0.2, random_state=42)
val_jp, test_jp, val_en, test_en = train_test_split(test_jp, test_en, test_size=0.5, random_state=42)

# 加载或训练 Word2Vec 模型
try:
    w2v_model = Word2Vec.load("word2vec_embeddings.model")
except FileNotFoundError:
    embedding_dim = 100
    w2v_model = Word2Vec(sentences=train_jp, vector_size=embedding_dim, window=5, min_count=1, sg=0)
    w2v_model.save("word2vec_embeddings.model")


# 数据预处理
import numpy as np

# 定义特殊符号
SOS_token = '<SOS>'
EOS_token = '<EOS>'
PAD_token = '<PAD>' # 批处理中对短句的填充
UNK_token = '<UNK>'

# 构建日文词汇表
jp_vocab = set(token for sentence in train_jp for token in sentence)    # 词汇表，是日文词的set
jp_vocab.update([SOS_token, EOS_token, PAD_token, UNK_token])
jp_word2idx = {word: idx for idx, word in enumerate(jp_vocab)}  # 具体的词和id的相互映射
jp_idx2word = {idx: word for word, idx in jp_word2idx.items()} 

# 构建英文词汇表
en_vocab = set(token for sentence in train_en for token in sentence)
en_vocab.update([SOS_token, EOS_token, PAD_token, UNK_token])
en_word2idx = {word: idx for idx, word in enumerate(en_vocab)}
en_idx2word = {idx: word for word, idx in en_word2idx.items()}

# 定义最大句子长度
max_len_jp = max(len(sentence) for sentence in train_jp) + 2  # 加上 <SOS> 和 <EOS>
max_len_en = max(len(sentence) for sentence in train_en) + 2

# 将句子转换为索引并进行填充 input: ['私', 'は', '元気', 'です'] output: [1, 2, 3, 4, 0, 0, 0]
def sentence_to_indices(sentence, word2idx, max_len):
    indices = [word2idx.get(token, word2idx[UNK_token]) for token in sentence] 
    indices = [word2idx[SOS_token]] + indices + [word2idx[EOS_token]]
    if len(indices) < max_len:
        indices += [word2idx[PAD_token]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

train_jp_indices = [sentence_to_indices(s, jp_word2idx, max_len_jp) for s in train_jp]
train_en_indices = [sentence_to_indices(s, en_word2idx, max_len_en) for s in train_en]

# 获取日文嵌入矩阵
embedding_dim = w2v_model.vector_size
jp_embedding_matrix = np.zeros((len(jp_vocab), embedding_dim))

for word, idx in jp_word2idx.items():
    if word in w2v_model.wv:
        jp_embedding_matrix[idx] = w2v_model.wv[word]
    else:
        jp_embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))

# 对于英文，嵌入层将在训练过程中学习




import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, jp_embedding_matrix):
        super(Seq2SeqModel, self).__init__()
        # 编码器嵌入层，使用预训练的日文嵌入
        self.encoder_embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder_embedding.weight.data.copy_(torch.from_numpy(jp_embedding_matrix))
        self.encoder_embedding.weight.requires_grad = False  # 冻结嵌入层

        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # 解码器嵌入层，随机初始化，需要训练
        self.decoder_embedding = nn.Embedding(output_dim, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.fc.out_features

        # 初始化解码器输出张量
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)

        # 编码器
        embedded_src = self.encoder_embedding(src)
        _, (hidden, cell) = self.encoder_lstm(embedded_src)

        # 解码器初始输入
        input = trg[:, 0]  # <SOS> 标记

        for t in range(1, trg_len):
            embedded_input = self.decoder_embedding(input).unsqueeze(1)
            output, (hidden, cell) = self.decoder_lstm(embedded_input, (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t] = prediction

            # 决定是否使用教师强制
            teacher_force = np.random.rand() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs




# 创建 TensorDataset（不然batch太大）
train_dataset = TensorDataset(torch.tensor(train_jp_indices, dtype=torch.long),
                              torch.tensor(train_en_indices, dtype=torch.long))

# 定义批次大小
batch_size = 32

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义模型参数
input_dim = len(jp_vocab)
output_dim = len(en_vocab)
hidden_dim = 256

model = Seq2SeqModel(input_dim, output_dim, embedding_dim, hidden_dim, jp_embedding_matrix)
criterion = nn.CrossEntropyLoss(ignore_index=en_word2idx[PAD_token])
optimizer = torch.optim.Adam(model.parameters())

# 开始训练
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for src_batch, trg_batch in train_loader:
        optimizer.zero_grad()
        output = model(src_batch, trg_batch)
        
        # 调整输出形状以计算损失
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg_batch[:, 1:].reshape(-1)
    
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')


# 保存模型
torch.save(model.state_dict(), 'model.pth')




def translate_sentence(sentence, model, jp_word2idx, en_idx2word, max_len_en):
    model.eval()
    tokens = mecab_tagger.parse(sentence).strip().split()
    indices = sentence_to_indices(tokens, jp_word2idx, max_len_jp)
    src_tensor = torch.tensor([indices], dtype=torch.long)

    with torch.no_grad():
        embedded_src = model.encoder_embedding(src_tensor)
        _, (hidden, cell) = model.encoder_lstm(embedded_src)
        input = torch.tensor([en_word2idx[SOS_token]], dtype=torch.long)

        outputs = []
        for _ in range(max_len_en):
            embedded_input = model.decoder_embedding(input).unsqueeze(0)
            output, (hidden, cell) = model.decoder_lstm(embedded_input, (hidden, cell))
            prediction = model.fc(output.squeeze(0))
            top1 = prediction.argmax(1)
            if top1.item() == en_word2idx[EOS_token]:
                break
            outputs.append(en_idx2word[top1.item()])
            input = top1

    return ' '.join(outputs)

# 测试
test_sentence = '昨日はお肉を食べません。'
translation = translate_sentence(test_sentence, model, jp_word2idx, en_idx2word, max_len_en)
print(f'原句: {test_sentence}')
print(f'翻译: {translation}')
