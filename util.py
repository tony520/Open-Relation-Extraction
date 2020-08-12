"""
Convert a list to a string
"""

def list_to_string(in_list):
    strg = ''
    strg = ' '.join([str(elem) for elem in in_list])
    return strg

"""
Load training data and all data
"""
def select_row_data(filename):
    all_sents = open(filename, "r")
    all_data_sentences = (all_sents.read()).split("\n")
    return all_data_sentences

def select_dataset(sentsname, tagsname):
    train_sents = open(sentsname, "r")
    training_sentences = (train_sents.read()).split("\n")
    train_tags = open(tagsname, "r")
    training_tags = (train_tags.read()).split("\n")
    return training_sentences, training_tags

"""
Load training data from row data files
"""
def load_train_dataset(training_sentences, training_tags):
	assert len(training_sentences) == len(training_tags)
    train_s = []
    train_t = []
    for i in range(len(training_sentences)):
        if len(training_sentences[i]) > 0:
            train_s.append(training_sentences[i])
            train_t.append(training_tags[i])
        else:
            continue
    
    assert len(train_s) == len(train_t)
    training_data = [(train_s[i].split(), train_t[i].split()) for i in range(len(train_t))]
    return training_data