from gensim.models import Word2Vec
# Load the trained model
model = Word2Vec.load("word2vec_embeddings.model")

# Get all words in the vocabulary
vocab_words = list(model.wv.index_to_key)
print(f"Total Vocabulary Size: {len(vocab_words)}")
print("First 10 words in the vocabulary:", vocab_words[-10:])

# Get the most similar words to a given word
word = "走れ"  # Japanese for "book"
try:
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}':")
    for similar_word, score in similar_words:
        print(f"  {similar_word} ({score:.4f})")
except KeyError:
    print(f"The word '{word}' was not found in the vocabulary.")

sentences = ['1 22 333','555 1 666']
print(sentence.split() for sentence in sentences)