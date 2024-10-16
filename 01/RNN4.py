# 卡在最后几个话的测试。跑通了，但发现结果根本不对
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

# 从训练集构建日文和英文的词汇表
from collections import Counter

def build_vocab(tokenized_sentences, min_freq=1):
    word_freq = Counter()
    for sentence in tokenized_sentences:
        word_freq.update(sentence)
    vocab = {word: idx + 4 for idx, (word, freq) in enumerate(word_freq.items()) if freq >= min_freq} # 从4开始编号
    vocab['<PAD>'] = 0  # 填充符
    vocab['<SOS>'] = 1  # 句子起始符
    vocab['<EOS>'] = 2  # 句子结束符
    vocab['<UNK>'] = 3  # 未知词
    return vocab

# 利用训练数据构建词汇表，包含了特殊符号。是词到索引的映射
jp_vocab = build_vocab(train_jp)
en_vocab = build_vocab(train_en)

# 创建索引到词的映射
jp_idx2word = {idx: word for word, idx in jp_vocab.items()}
en_idx2word = {idx: word for word, idx in en_vocab.items()}

# 将句子（词列表）转换为索引序列
def sentence_to_indices(sentence, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in sentence]   # 未登录词用<UNK>表示

# 定义数据集类，如TranslationDataset(train_jp, train_en, jp_vocab, en_vocab)
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab, max_len=50):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.source_sentences)

    # 返回源句子、目标输入和目标输出的索引序列
    def __getitem__(self, idx):
        src_indices = sentence_to_indices(self.source_sentences[idx], self.source_vocab)
        tgt_indices = sentence_to_indices(self.target_sentences[idx], self.target_vocab)
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len - 2]
        # 添加 <SOS> 和 <EOS>
        tgt_input = [self.target_vocab['<SOS>']] + tgt_indices
        tgt_output = tgt_indices + [self.target_vocab['<EOS>']]
        return torch.tensor(src_indices), torch.tensor(tgt_input), torch.tensor(tgt_output)

# 定义批处理函数，进行填充
def collate_fn(batch):
    src_batch, tgt_input_batch, tgt_output_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_input_batch]

    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=jp_vocab['<PAD>'], batch_first=True)
    tgt_input_padded = nn.utils.rnn.pad_sequence(tgt_input_batch, padding_value=en_vocab['<PAD>'], batch_first=True)
    tgt_output_padded = nn.utils.rnn.pad_sequence(tgt_output_batch, padding_value=en_vocab['<PAD>'], batch_first=True)

    return src_padded.to(device), torch.tensor(src_lens).to(device), \
           tgt_input_padded.to(device), tgt_output_padded.to(device), torch.tensor(tgt_lens).to(device)

# 创建数据集和数据加载器
batch_size = 64

train_dataset = TranslationDataset(train_jp, train_en, jp_vocab, en_vocab)
val_dataset = TranslationDataset(val_jp, val_en, jp_vocab, en_vocab)
test_dataset = TranslationDataset(test_jp, test_en, jp_vocab, en_vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 加载或训练 Word2Vec 模型
try:
    w2v_model = Word2Vec.load("word2vec_embeddings.model")
except FileNotFoundError:
    embedding_dim = 100
    w2v_model = Word2Vec(sentences=train_jp, vector_size=embedding_dim, window=5, min_count=1, sg=0, compute_loss=True)
    print(w2v_model.get_latest_training_loss())
    w2v_model.save("word2vec_embeddings.model")

# 测试W2V
word = "彼"
try:
    similar_words = w2v_model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}':")
    for similar_word, score in similar_words:
        print(f"  {similar_word} ({score:.4f})")
except KeyError:
    print(f"The word '{word}' was not found in the vocabulary.")

# 准备日文的嵌入矩阵。对吗？idx+4之后
embedding_dim = w2v_model.vector_size
jp_embedding_matrix = torch.zeros(len(jp_vocab), embedding_dim)
for word, idx in jp_vocab.items():
    if word in w2v_model.wv:
        jp_embedding_matrix[idx] = torch.tensor(w2v_model.wv[word])
    else:
        jp_embedding_matrix[idx] = torch.zeros(embedding_dim)

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, jp_embedding_matrix):
        super(Seq2Seq, self).__init__()
        # Encoder
        self.encoder_embedding = nn.Embedding.from_pretrained(jp_embedding_matrix, freeze=False, padding_idx=jp_vocab['<PAD>'])
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Decoder
        self.decoder_embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=en_vocab['<PAD>'])
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 设置Sos和Eos的token
        self.sos_token = en_vocab['<SOS>']
        self.eos_token = en_vocab['<EOS>']

    def forward(self, src, src_lengths, tgt_input=None, tgt_lengths=None, max_len=50):
        # Encoder
        embedded_src = self.encoder_embedding(src)
        packed_src = nn.utils.rnn.pack_padded_sequence(embedded_src, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(packed_src)

        # 如果没有提供目标输入，进入推理模式
        if tgt_input is None:
            # 推理模式逐步生成翻译
            batch_size = src.size(0)
            outputs = []
            input_token = torch.tensor([self.sos_token] * batch_size).unsqueeze(1).to(src.device)  # 初始输入为 <SOS>
            
            for t in range(max_len):
                embedded_tgt = self.decoder_embedding(input_token)
                decoder_output, (hidden, cell) = self.decoder_lstm(embedded_tgt, (hidden, cell))
                prediction = self.fc_out(decoder_output.squeeze(1))
                top1 = prediction.argmax(1).unsqueeze(1)  # 取出概率最高的词
                outputs.append(top1)
                input_token = top1  # 下一步输入是当前预测的词
                
                # 如果预测到 <EOS>，则停止生成
                if (top1 == self.eos_token).all():
                    break
            
            outputs = torch.cat(outputs, dim=1)
            return outputs  # 返回生成的句子索引

        else:
            # 训练模式，使用目标输入进行解码
            embedded_tgt = self.decoder_embedding(tgt_input)
            packed_tgt = nn.utils.rnn.pack_padded_sequence(embedded_tgt, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False)
            decoder_outputs, _ = self.decoder_lstm(packed_tgt, (hidden, cell))
            outputs, _ = nn.utils.rnn.pad_packed_sequence(decoder_outputs, batch_first=True, total_length=tgt_input.size(1))
            predictions = self.fc_out(outputs)
            return predictions

# 初始化模型、损失函数和优化器
input_dim = len(jp_vocab)
output_dim = len(en_vocab)
hidden_dim = 256

model = Seq2Seq(input_dim, output_dim, embedding_dim, hidden_dim, jp_embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)




import torch
import math
from nltk.translate.bleu_score import sentence_bleu

# 使用 NLTK 计算 BLEU 分数
def calculate_bleu(model, test_loader, en_vocab):
    model.eval()
    candidate_sentences = []
    reference_sentences = []
    
    # 从索引到单词的映射表
    idx_to_word = {idx: word for word, idx in en_vocab.items()}
    
    with torch.no_grad():
        for src_batch, src_lens, tgt_input_batch, tgt_output_batch, tgt_lens in test_loader:
            output = model(src_batch, src_lens, tgt_input_batch, tgt_lens)
            # 获取最大概率的单词
            predicted_tokens = output.argmax(-1).cpu().tolist()
            reference_tokens = tgt_output_batch.cpu().tolist()
            
            for pred, ref in zip(predicted_tokens, reference_tokens):
                # 去除特殊标记 <PAD>, <SOS>, <EOS>
                pred_sentence = [idx_to_word[idx] for idx in pred if idx not in [en_vocab['<PAD>'], en_vocab['<SOS>'], en_vocab['<EOS>']]]
                ref_sentence = [idx_to_word[idx] for idx in ref if idx not in [en_vocab['<PAD>'], en_vocab['<SOS>'], en_vocab['<EOS>']]]
                
                candidate_sentences.append(pred_sentence)
                reference_sentences.append([ref_sentence])  # 参考翻译应该是列表的列表形式

    # 计算每个句子的 BLEU 分数并平均
    total_bleu_score = 0
    for candidate, reference in zip(candidate_sentences, reference_sentences):
        bleu_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))  # 4-gram BLEU
        total_bleu_score += bleu_score

    avg_bleu_score = total_bleu_score / len(candidate_sentences)
    return avg_bleu_score

# 定义一个函数来计算Perplexity
def calculate_perplexity(model, test_loader, criterion, output_dim):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src_batch, src_lens, tgt_input_batch, tgt_output_batch, tgt_lens in test_loader:
            output = model(src_batch, src_lens, tgt_input_batch, tgt_lens)
            # 调整形状以适配CrossEntropyLoss
            output = output.view(-1, output_dim)
            tgt_output = tgt_output_batch.view(-1)
            loss = criterion(output, tgt_output)
            
            total_loss += loss.item() * tgt_output.size(0)  # 累积每个词的损失
            total_tokens += tgt_output.size(0)  # 统计总词数
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity




# 训练模型
num_epochs = 15
model_save_path = "seq2seq_model.pth"

try:
    model.load_state_dict(torch.load(model_save_path))
except FileNotFoundError:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src_batch, src_lens, tgt_input_batch, tgt_output_batch, tgt_lens in train_loader:
            optimizer.zero_grad()
            output = model(src_batch, src_lens, tgt_input_batch, tgt_lens)
            # 调整形状以适配 CrossEntropyLoss
            output = output.view(-1, output_dim)
            tgt_output = tgt_output_batch.view(-1)
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 在验证集上评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_batch, src_lens, tgt_input_batch, tgt_output_batch, tgt_lens in val_loader:
                output = model(src_batch, src_lens, tgt_input_batch, tgt_lens)
                output = output.view(-1, output_dim)
                tgt_output = tgt_output_batch.view(-1)
                loss = criterion(output, tgt_output)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # 在训练集上评估 BLEU 和 Perplexity
        bleu_score = calculate_bleu(model, train_loader, en_vocab)
        perplexity = calculate_perplexity(model, train_loader, criterion, output_dim)
        print(f"BLEU Score on Train Set: {bleu_score:.4f}")
        print(f"Perplexity on Train Set: {perplexity:.4f}")

        # 在验证集上评估 BLEU 和 Perplexity
        bleu_score = calculate_bleu(model, val_loader, en_vocab)
        perplexity = calculate_perplexity(model, val_loader, criterion, output_dim)
        print(f"BLEU Score on Validation Set: {bleu_score:.4f}")
        print(f"Perplexity on Validation Set: {perplexity:.4f}")

    # 保存模型
    torch.save(model.state_dict(), model_save_path)




# 在训练集上评估 BLEU 和 Perplexity
bleu_score = calculate_bleu(model, train_loader, en_vocab)
perplexity = calculate_perplexity(model, train_loader, criterion, output_dim)
print(f"BLEU Score on Train Set: {bleu_score:.4f}")
print(f"Perplexity on Train Set: {perplexity:.4f}")

# 在验证集上评估 BLEU 和 Perplexity
bleu_score = calculate_bleu(model, val_loader, en_vocab)
perplexity = calculate_perplexity(model, val_loader, criterion, output_dim)
print(f"BLEU Score on Validation Set: {bleu_score:.4f}")
print(f"Perplexity on Validation Set: {perplexity:.4f}")

# 在测试集上评估 BLEU 和 Perplexity
bleu_score = calculate_bleu(model, test_loader, en_vocab)
perplexity = calculate_perplexity(model, test_loader, criterion, output_dim)
print(f"BLEU Score on Test Set: {bleu_score:.4f}")
print(f"Perplexity on Test Set: {perplexity:.4f}")




def translate_sentence(sentence, model, jp_vocab, en_vocab, device):
    # 分词并将句子转换为索引
    tokenized_sentence = [jp_vocab.get(word, jp_vocab['<UNK>']) for word in sentence]  # 用 <UNK> 处理未登录词
    src_tensor = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0).to(device)  # 添加 batch 维度
    
    src_len = torch.tensor([len(tokenized_sentence)]).to(device)  # 句子的长度
    
    # 模型生成翻译
    model.eval()
    with torch.no_grad():
        output = model(src_tensor, src_len)  # 只提供 src 和 src_len，tgt_input 留空
        output = output.squeeze(0).cpu().tolist()  # 移除 batch 维度
    
    # 转换预测结果为实际单词
    translated_sentence = [en_idx2word.get(idx, '<UNK>') for idx in output if idx not in [en_vocab['<PAD>'], en_vocab['<SOS>'], en_vocab['<EOS>']]]
    
    return ' '.join(translated_sentence)

# 测试模型对给定句子的翻译
case_1 = "私の名前は愛です"
case_2 = "昨日はお肉を食べません"
case_3 = "いただきますよう"
case_4 = "秋は好きです"
case_5 = "おはようございます"

test_cases = [case_1, case_2, case_3, case_4, case_5]



for i, case in enumerate(test_cases, 1):
    # 将句子转换为字符列表 (假设已经进行了必要的分词，或根据需要进行分词)
    tokenized_case = list(case)
    translated_sentence = translate_sentence(tokenized_case, model, jp_vocab, en_vocab, device)
    print(f"Case {i}: {case}")
    print(f"Translated: {translated_sentence}")
    print()
