import sys
import collections

import numpy as np
import pandas as pd

def fetch_data(train_file='sentences.train', valid_file='sentences.eval'):

    f_train = open('data/%s' %train_file, 'r')
    f_valid = open('data/%s' %valid_file, 'r')

    train = f_train.readlines()
    valid = f_valid.readlines()

    f_train.close()
    f_valid.close()

    return (train, valid)

def construct_vocab(corpus, vocab_size=20000,, sentence_len=30 base_vocab={'<unk>': 0, '<bos>': 1, '<eos>': 2, '<pad>': 3}):
    """
    Associate each word in the vocabulary with a (unique) ID

    Parameters:
    -----------
    corpus : list 
        Each entry in the list corresponds to a sentence in the corpus

    vocab_size : int
        The size of the vocabulary

    base_vocab: dict
        Initialization for the vocabulary 

    Returns: 
    --------
    vocab : dict 
        The vocabulary dictionary where keys correspond to (unique) words in the vocabulary and the values correspond to the unique ID of the word
    """
    counter = collections.Counter()

    for line in corpus:
        sentence = line.strip().split(" ") 

        # Ignore senteces longer than sentence_len - 2
        if len(sentence) > sentence_len - 2: 
            continue

        counter.update(sentence)

    # Keep the vocab_size - (base_vocab size) most common words from the corpus
    most_common = counter.most_common(vocab_size - len(base_vocab))

    # Initialize the vocabulary 
    vocab = dict(base_vocab)

    # Associate each word in the vocabulary with a unique ID number
    ID = len(base_vocab)

    for token, _ in most_common:
        vocab[token] = ID
        ID += 1

    return vocab

def encode_text(corpus, vocab, sentence_len=30): 
    """
    Encode words in the text in terms of their ID in the vocabulary. Sentences that are longer than 30 tokens (including <bos> and <eos> are ignored).
    Parameters: 
    -----------
    corpus : list 
        Each entry in the list corresponds to a sentence in the corpus

    vocab : dict
        The vocabulary dictionary

    sentence_len : int
        The (maximal) length of each sentence

    Returns:
    --------
    data : array-like, shape (n_sentences, sentence_len)
        Each row corresponds to a sentence. Entries in a row are integers and correspond to the vocabulary word ID.

    """

    # Initialize the data matrix
    data = np.full(shape=(len(corpus), sentence_len), fill_value=vocab['<pad>'], dtype=int)

    data[:,0] = vocab['<bos>'] # Beggining of sentence  

    # Fill-in the rest of the data matrix

    s_ID = 0 # Sentence ID
    long_count = 0 # Number of long sentences

    for line in corpus:
        sentence = line.strip().split(" ")

        # Ignore senteces longer than sentence_len - 2
        if len(sentence) > sentence_len - 2: 
            long_count += 1
            continue

        t_ID = 1 # Word ID

        for token in sentence: 

            if token in vocab.keys():
                data[s_ID, t_ID] = vocab[token] # Within-vocabulary
            else:
                data[s_ID, t_ID] = vocab['<unk>'] # Out-of-vocabulary

            t_ID += 1

        data[s_ID, t_ID] = vocab['<eos>'] # End of sentence

        s_ID += 1

    print("Total number of sentences with length greater than 28 tokens: ", long_count)

    # Remove rows that correspond to ignored sentences
    data = data[:-long_count,:]

    return data

if __name__ == '__main__':

    train_corpus, valid_corpus = fetch_data()

    encode_text(corpus=train_corpus, vocab=construct_vocab(train_corpus))
