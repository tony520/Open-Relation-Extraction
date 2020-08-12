import bcolz as bcolz
import pickle as pickle
import numpy as np

"""
Load sentences from data file
"""
def select_row_data(filename):
    all_sents = open(filename, "r")
    all_data_sentences = (all_sents.read()).split("\n")
    return all_data_sentences

"""
Save GloVe pretrained word embeddings 50 dimensions
"""
def save_pretrained_word_embeddings_glove50d():
	words = []
	idx = 0
	word_to_ix = {}
	vectors = bcolz.carray(np.zeros(1), rootdir = f'GloVe/6B.50.dat', mode = 'w')

	with open(f'GloVe/glove.6B.50d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word_to_ix[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir = f'GloVe/6B.50.dat', mode = 'w')
	vectors.flush()
	pickle.dump(words, open(f'GloVe/6B.50_words.pkl', 'wb'))
	pickle.dump(word_to_ix, open(f'GloVe/6B.50_word_idx.pkl', 'wb'))

"""
Save GloVe pretrained word embeddings 100 dimensions
"""
def save_pretrained_word_embeddings_glove100d():
	words = []
	idx = 0
	word_to_ix = {}
	vectors = bcolz.carray(np.zeros(1), rootdir = f'GloVe/6B.100.dat', mode = 'w')

	with open(f'GloVe/glove.6B.100d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word_to_ix[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir = f'GloVe/6B.100.dat', mode = 'w')
	vectors.flush()
	pickle.dump(words, open(f'GloVe/6B.100_words.pkl', 'wb'))
	pickle.dump(word_to_ix, open(f'GloVe/6B.100_word_idx.pkl', 'wb'))

"""
Save GloVe pretrained word embeddings 200 dimensions
"""
def save_pretrained_word_embeddings_glove200d():
	words = []
	idx = 0
	word_to_ix = {}
	vectors = bcolz.carray(np.zeros(1), rootdir = f'GloVe/6B.200.dat', mode = 'w')

	with open(f'GloVe/glove.6B.200d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word_to_ix[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir = f'GloVe/6B.200.dat', mode = 'w')
	vectors.flush()
	pickle.dump(words, open(f'GloVe/6B.200_words.pkl', 'wb'))
	pickle.dump(word_to_ix, open(f'GloVe/6B.200_word_idx.pkl', 'wb'))

"""
Save GloVe pretrained word embeddings 300 dimensions
"""
def save_pretrained_word_embeddings_glove300d():
	words = []
	idx = 0
	word_to_ix = {}
	vectors = bcolz.carray(np.zeros(1), rootdir = f'GloVe/6B.300.dat', mode = 'w')

	with open(f'GloVe/glove.6B.300d.txt', 'rb') as f:
	    for l in f:
	        line = l.decode().split()
	        word = line[0]
	        words.append(word)
	        word_to_ix[word] = idx
	        idx += 1
	        vect = np.array(line[1:]).astype(np.float)
	        vectors.append(vect)

	vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir = f'GloVe/6B.300.dat', mode = 'w')
	vectors.flush()
	pickle.dump(words, open(f'GloVe/6B.300_words.pkl', 'wb'))
	pickle.dump(word_to_ix, open(f'GloVe/6B.300_word_idx.pkl', 'wb'))


"""
Generate the dictionary of (word, pretrained word embedding vectors)
Provide word embedding for different dimensions (50, 100, 200, 300)
"""
def generate_glove_dict(dimension):
	if dimension == 50:
		save_pretrained_word_embeddings_glove50d()
		vectors = bcolz.open(f'GloVe/6B.50.dat')[:]
		words = pickle.load(open(f'GloVe/6B.50_words.pkl', 'rb'))
		word_to_ix = pickle.load(open(f'GloVe/6B.50_word_idx.pkl', 'rb'))
		glove = {w: vectors[word_to_ix[w]] for w in words}
		print("Example pretrained vector for word (the) in 50d: ", glove['the'])
		return glove
	if dimension == 100:
		save_pretrained_word_embeddings_glove100d()
		vectors = bcolz.open(f'GloVe/6B.100.dat')[:]
		words = pickle.load(open(f'GloVe/6B.100_words.pkl', 'rb'))
		word_to_ix = pickle.load(open(f'GloVe/6B.100_word_idx.pkl', 'rb'))
		glove = {w: vectors[word_to_ix[w]] for w in words}
		print("Example pretrained vector for word (the) in 100d: ", glove['the'])
		return glove
	if dimension == 200:
		save_pretrained_word_embeddings_glove200d()
		vectors = bcolz.open(f'GloVe/6B.200.dat')[:]
		words = pickle.load(open(f'GloVe/6B.200_words.pkl', 'rb'))
		word_to_ix = pickle.load(open(f'GloVe/6B.200_word_idx.pkl', 'rb'))
		glove = {w: vectors[word_to_ix[w]] for w in words}
		print("Example pretrained vector for word (the) in 200d: ", glove['the'])
		return glove
	if dimension == 300:
		save_pretrained_word_embeddings_glove300d()
		vectors = bcolz.open(f'GloVe/6B.300.dat')[:]
		words = pickle.load(open(f'GloVe/6B.300_words.pkl', 'rb'))
		word_to_ix = pickle.load(open(f'GloVe/6B.300_word_idx.pkl', 'rb'))
		glove = {w: vectors[word_to_ix[w]] for w in words}
		print("Example pretrained vector for word (the) in 300d: ", glove['the'])
		return glove

"""
Create the weight matrix for the BiLSTM-CRF model
Add words not in GloVe but in the training sentences into the weight matrix
Assign a random embedding vector to each of non-GloVe word
"""
def create_weight_matrix(filepath, dimension):
	glove = generate_glove_dict(dimension)
	all_data_sentences = select_row_data(filepath)
	all_data = [all_data_sentences[i].split() for i in range(len(all_data_sentences))]
	for sentence in all_data:
	    for word in sentence:
	        if word not in words:
	            words.append(word)
	            word_to_ix[word] = idx
	            idx += 1
            

	matrix_len = len(words)
	weights_matrix = np.zeros((matrix_len, 50))
	words_found = 0

	print("Total words: ", len(words))

	for i, word in enumerate(words):
	    try:
	        weights_matrix[i] = glove[word]
	        words_found += 1
	    except KeyError:
	        weights_matrix[i] = np.random.normal(scale = 0.6, size = (50, ))

	print("Number of new word: ", words_found)
	return weights_matrix
