# Open-Relation-Extraction
Supervised BiLSTM-CRF model for Open Relation Extraction

Table of Contents
=================

* [Dataset](/data)
* [Pre-trained Word Embeddings (GloVe)](/pretrained_word_embeddings)
* [BiLSTM-CRF Model](/models)
* [Evaluation Scripts](/evaluations)
* [New Tagging Schema](/new_tagging_schema)
* [Benchmark](/benckmark)

## Requirements

Install `PyTorch` from `pip`:
```bash
> pip install torch
```

Install `Pickle` from `pip` for pre-trained word embeddings:
```bash
> pip install pickle
```

## Using Pre-trained Word Embedding (GloVe)
In this project, we use GloVe 50 for pre-trained word embedding. The launching scripts are in folder `pretrained_word_embeddings`. Following the next few steps to use.
1. Download the GloVe word embeddings using `download_unzip_GloVe.py`. It may takes few minutes.
```bash
> python3 download_unzip_GloVe.py
```
2. Load pre-trained word embeddings and creat the weight matrix for BiLSTM-CRFmodel.
```bash
> python3 load_pretrained_word_embeddings.py
```

## Dataset
We use training data from [Stanovsky dataset](https://github.com/gabrielStanovsky/supervised-oie/tree/master/data). We convert the sentences and tags into `.txt` format for our model training. The input sequence and tagging sequence formats are like:
```
courtaulds ' spinoff reflects pressure on british industry to boost share prices beyond the reach of corporate raiders
   A0-B  A0-I A0-I     P-B      A1-B  A1-I  A1-I    A1-I   O   O     O      O      O     O    O    O     O        O
```
Use `preprocessing.py` to convert data from `.conll` to `.txt`. The data and converting script are in folder `data`
```bash
> python3 preprocessing.py
```

Testing data sets also come from the repository of [supervised-oie](https://github.com/gabrielStanovsky/supervised-oie/tree/master/data).

## Model Training
In this project, we build a Supervised BiLSTM-CRF model for this sequence tagging task. Then we extract relations from the output tagging sequence. We take the basic BiLSTM-CRF architecture from [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html) as a reference. The code of our Supervised Open RE model is in the folder `models`. To train the model, set the number of epoch (default: 50) and hyperparameters (default learning rate: 0.01) then run:
```bash
python3 train.py
```
After the training process we can get the trained model and use it for relation extraction from sentences.

## Apply New Tagging Schema
To use the new tagging schema, go to the folder `new_tagging_schema`.
