import numpy as np
import os
import MeCab
import nltk
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Embedding, Dense

# 读取数据
with open('eng_jpn.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 分割日文和英文句子
jp_sentences = [line.split('\t')[0] for line in lines]
en_sentences = [line.split('\t')[1].strip() for line in lines]

# 初始化分词器
mecab_tagger = MeCab.Tagger('-O wakati')
nltk.download('punkt')

# 对句子进行分词
tokenized_jp = [mecab_tagger.parse(sentence).strip().split() for sentence in jp_sentences]
tokenized_en = [nltk.word_tokenize(sentence.lower()) for sentence in en_sentences]

# 添加开始和结束标记到英文句子
tokenized_en = [['<sos>'] + sentence + ['<eos>'] for sentence in tokenized_en]

# 划分数据集
train_jp, test_jp, train_en, test_en = train_test_split(
    tokenized_jp, tokenized_en, test_size=0.2, random_state=42)
val_jp, test_jp, val_en, test_en = train_test_split(
    test_jp, test_en, test_size=0.5, random_state=42)

# 训练或加载预训练的 Word2Vec 模型
embedding_dim = 100
try:
    w2v_model = Word2Vec.load("word2vec_embeddings.model")
except FileNotFoundError:
    w2v_model = Word2Vec(
        sentences=train_jp, vector_size=embedding_dim, window=5, min_count=1, sg=0)
    w2v_model.save("word2vec_embeddings.model")

# 为日文创建词汇表
jp_tokenizer = Tokenizer()
jp_tokenizer.fit_on_texts(train_jp)
jp_vocab_size = len(jp_tokenizer.word_index) + 1

# 为英文创建词汇表（包括特殊标记）
en_tokenizer = Tokenizer(filters='')
en_tokenizer.fit_on_texts(train_en)
en_vocab_size = len(en_tokenizer.word_index) + 1

# 将句子转换为序列
train_jp_seq = jp_tokenizer.texts_to_sequences(train_jp)
val_jp_seq = jp_tokenizer.texts_to_sequences(val_jp)
test_jp_seq = jp_tokenizer.texts_to_sequences(test_jp)

train_en_seq = en_tokenizer.texts_to_sequences(train_en)
val_en_seq = en_tokenizer.texts_to_sequences(val_en)
test_en_seq = en_tokenizer.texts_to_sequences(test_en)

# 获取序列的最大长度
max_jp_len = max(len(seq) for seq in train_jp_seq)
max_en_len = max(len(seq) for seq in train_en_seq)

# 对序列进行填充
train_jp_pad = pad_sequences(train_jp_seq, maxlen=max_jp_len, padding='post')
val_jp_pad = pad_sequences(val_jp_seq, maxlen=max_jp_len, padding='post')
test_jp_pad = pad_sequences(test_jp_seq, maxlen=max_jp_len, padding='post')

train_en_pad = pad_sequences(train_en_seq, maxlen=max_en_len, padding='post')
val_en_pad = pad_sequences(val_en_seq, maxlen=max_en_len, padding='post')
test_en_pad = pad_sequences(test_en_seq, maxlen=max_en_len, padding='post')

# 准备解码器的输入和输出
def create_decoder_sequences(sequences, max_length):
    decoder_input_data = []
    decoder_target_data = []
    for seq in sequences:
        decoder_input_data.append(seq[:-1])
        decoder_target_data.append(seq[1:])
    decoder_input_padded = pad_sequences(
        decoder_input_data, maxlen=max_length-1, padding='post')
    decoder_target_padded = pad_sequences(
        decoder_target_data, maxlen=max_length-1, padding='post')
    return decoder_input_padded, decoder_target_padded

train_en_input, train_en_output = create_decoder_sequences(
    train_en_pad, max_en_len)
val_en_input, val_en_output = create_decoder_sequences(val_en_pad, max_en_len)
test_en_input, test_en_output = create_decoder_sequences(
    test_en_pad, max_en_len)

# 创建嵌入矩阵
embedding_matrix = np.zeros((jp_vocab_size, embedding_dim))
for word, i in jp_tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_vector = w2v_model.wv[word]
        embedding_matrix[i] = embedding_vector

# 定义模型路径
model_path = 'translation_model.h5'

if os.path.exists(model_path):
    print('正在加载已有的模型...')
    model = load_model(model_path)
else:
    print('未找到模型，正在构建和训练新模型...')
    # 编码器
    encoder_inputs = Input(shape=(max_jp_len,))
    encoder_embedding = Embedding(input_dim=jp_vocab_size, output_dim=embedding_dim,
                                  weights=[embedding_matrix], input_length=max_jp_len, trainable=False)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(256, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # 解码器
    decoder_inputs = Input(shape=(max_en_len-1,))
    decoder_embedding_layer = Embedding(input_dim=en_vocab_size, output_dim=256)
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm, _, _ = LSTM(256, return_sequences=True, return_state=True)(
        decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(en_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_lstm)

    # 定义模型
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 调整输出的形状以适应 sparse_categorical_crossentropy
    train_en_output = np.expand_dims(train_en_output, -1)
    val_en_output = np.expand_dims(val_en_output, -1)

    # 训练模型
    model.fit([train_jp_pad, train_en_input], train_en_output,
              batch_size=64,
              epochs=10,
              validation_data=([val_jp_pad, val_en_input], val_en_output))

    # 保存模型
    model.save(model_path)

# 如果模型是新加载的，我们需要重新定义推理模型
# 编码器模型用于推理
encoder_model = Model(encoder_inputs, encoder_states)

# 解码器的输入
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_embedding_inf = decoder_embedding_layer(decoder_inputs_single)

# 解码器 LSTM 层
decoder_lstm_inf, state_h_inf, state_c_inf = LSTM(256, return_sequences=True, return_state=True)(
    decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_lstm_inf)

# 解码器模型用于推理
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf)

# 定义解码函数
def decode_sequence(input_seq):
    # 编码输入句子得到状态向量
    states_value = encoder_model.predict(input_seq)

    # 生成长度为1的目标序列，初始化为 <sos>
    target_seq = np.array([[en_tokenizer.word_index['<sos>']]])

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 采样一个词
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = en_tokenizer.index_word.get(sampled_token_index, '')

        # 结束条件：达到最大长度或找到 <eos>
        if (sampled_word == '<eos>' or len(decoded_sentence.split()) > max_en_len):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        # 更新目标序列
        target_seq = np.array([[sampled_token_index]])

        # 更新状态
        states_value = [h, c]

    return decoded_sentence.strip()

# 定义翻译函数
def translate_sentence(sentence):
    # 对句子进行分词
    tokenized_sentence = mecab_tagger.parse(sentence).strip().split()
    # 转换为序列
    seq = jp_tokenizer.texts_to_sequences([tokenized_sentence])
    # 填充序列
    seq_padded = pad_sequences(seq, maxlen=max_jp_len, padding='post')
    # 翻译句子
    translated_sentence = decode_sequence(seq_padded)
    return translated_sentence

# 测试示例
test_sentence = '行け。'  # "Go."
print('输入句子:', test_sentence)
print('翻译结果:', translate_sentence(test_sentence))
