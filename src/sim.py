#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np 
import tensorflow as tf

# Custom dependencies 
import model 
import preprocessing 

if __name__ == '__main__':  

    train_corpus, valid_corpus, test_corpus = preprocessing.fetch_data()

    # Construct Vocabulary
    vocab = preprocessing.construct_vocab(train_corpus)

    # Encode words in the corpus in terms of their IDs
    train_data = preprocessing.encode_text(corpus=train_corpus, vocab=vocab)
    valid_data = preprocessing.encode_text(corpus=valid_corpus, vocab=vocab)
    test_data  = preprocessing.encode_text(corpus=test_corpus,  vocab=vocab)

    print('Training data matrix dimensions: ', train_data.shape)
    print('Validation data matrix dimensions: ', valid_data.shape)
    print('Testing data matrix dimensions: ', test_data.shape)

    ############################################################################

    #                              Experiment A

    ############################################################################
    model_A = model.LanguageModel(vocab=vocab, experiment_type='A')

    print('Total number of parameters: ', model_A.total_params())

    model_A.fit(train_data, val_data=valid_data, epochs=4)

    # Testing 
    test_res = model_A.evaluate(test_data)

    with open('test.perplexityΑ', 'w') as f:
        for res in test_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    ############################################################################

    #                              Experiment B

    ############################################################################
    model_B = model.LanguageModel(vocab=vocab, experiment_type='B')

    print('Total number of parameters: ', model_B.total_params())

    model_B.fit(train_data, val_data=valid_data, epochs=4)

    # Testing 
    test_res = model_B.evaluate(test_data)

    with open('test_perplexityB', 'w') as f:
        for res in test_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    ############################################################################

    #                              Experiment C

    ############################################################################
    model_C = model.LanguageModel(vocab=vocab, experiment_type='C', sentence_len=30, 
                embedding_dim=100, hidden_size=1024)

    print('Total number of parameters: ', model_C.total_params())

    model_C.fit(train_data, val_data=valid_data, epochs=4)

    # Testing 
    test_res = model_C.evaluate(test_data)

    with open('test.perplexityC', 'w') as f:
        for res in test_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()
