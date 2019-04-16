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
    model = model.LanguageModel(vocab=vocab, experiment_type='A')

    print('Total number of parameters: ', model.total_params())

    model.fit(train_data, val_data=valid_data)

    # Validation 
    val_res = model.evaluate(valid_data)

    with open('validation_perplexity_Α', 'w') as f:
        for res in val_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    # Testing 
    test_res = model.evaluate(test_data)

    with open('test_perplexity_Α', 'w') as f:
        for res in test_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    ############################################################################

    #                              Experiment B

    ############################################################################
    model = model.LanguageModel(vocab=vocab, experiment_type='B')

    print('Total number of parameters: ', model.total_params())

    model.fit(train_data, val_data=valid_data)

    # Validation 
    val_res = model.evaluate(valid_data)

    with open('validation_perplexity_B', 'w') as f:
        for res in val_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    # Testing 
    test_res = model.evaluate(test_data)

    with open('test_perplexity_B', 'w') as f:
        for res in test_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    ############################################################################

    #                              Experiment C

    ############################################################################
    model = model.LanguageModel(vocab=vocab, experiment_type='C', sentence_len=30, 
                embedding_dim=100, hidden_size=1024)

    print('Total number of parameters: ', model.total_params())

    model.fit(train_data, val_data=valid_data)

    # Validation 
    val_res = model.evaluate(valid_data)

    with open('validation_perplexity_C', 'w') as f:
        for res in val_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()

    # Testing 
    test_res = model.evaluate(test_data)

    with open('test_perplexity_C', 'w') as f:
        for res in test_res['perp_per_sent']:
            f.write("%s\n" % res)

    f.close()
