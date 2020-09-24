import chars2vec


dim = 50

path_to_model = './'

words = ['list', 'of', 'words']

# Load pretrained model, create word embeddings
c2v_model = chars2vec.load_model(path_to_model)
word_embeddings = c2v_model.vectorize_words(words)

print(word_embeddings.shape)
print(word_embeddings[0])
