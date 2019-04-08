import numpy as np 
import tensorflow as tf

# Custom dependencies 
import model 
import preprocessing 

if __name__ == '__main__':  

    train_corpus, valid_corpus = preprocessing.fetch_data()

    # Construct Vocabulary
    vocab = preprocessing.construct_vocab(train_corpus)

    # Encode words in the corpus in terms of their IDs
    train_data = preprocessing.encode_text(corpus=train_corpus, vocab=vocab)
    valid_data = preprocessing.encode_text(corpus=valid_corpus, vocab=vocab)

    print('Training data matrix dimensions: ', train_data.shape)
    print('Validation data matrix dimensions: ', valid_data.shape)

    model = model.LanguageModel(vocab=vocab)

    model.fit(train_data)

    model.evaluate(valid_data)
