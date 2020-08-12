import pandas as pd
from util import list_to_string

def convert_conll_to_txt(IN_FILE, OUT_FILE_SENTS, OUT_FILE_TAGS):
	sents_file = open(OUT_FILE_SENTS, 'w')
	tags_file = open(OUT_FILE_TAGS, 'w')

	sents = []
	tags = []
	tags_dict = []
	temp_sents = []
	temp_tags = []

	f = open(IN_FILE, 'r')
	lines = f.readlines()
	for line in lines:
		if line == '\n':
			sents.append(list_to_string(temp_sents).lower())
			tags.append(list_to_string(temp_tags))
			temp_sents = []
			temp_tags = []
		else:
			temp_tp = line.split(' ')
			temp_sents.append(temp_tp[0])
			temp_tags.append(temp_tp[-1].strip('\n'))
			if temp_tp[-1].strip('\n') not in tags_dict:
				tags_dict.append(temp_tp[-1].strip('\n'))

	assert len(sents) == len(tags)
	for i in range(len(sents)):
		sents_file.write(sents[i] + '\n')
		tags_file.write(tags[i] + '\n')

	sents_file.close()
	tags_file.close()
	print("Done. The training data (.txt) have been saved in data folder")


convert_conll_to_txt("data/train.conll", "data/train.sents.txt", "data/train.tags.txt")
convert_conll_to_txt("data/val.conll", "data/val.sents.txt", "data/val.tags.txt")
convert_conll_to_txt("data/train+val.conll", "data/train+val.sents.txt", "data/train+val.tags.txt")