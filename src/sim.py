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

    print('Total number of parameters: ', model.total_params())

    model.fit(train_data, val_data=valid_data)

    val_res = model.evaluate(valid_data)

    print('From evaluation method')

    print('Accumulated validation loss: ', val_res['loss'])

    with open('validation.perplexity', 'w') as f:
        for res in val_res['perp_per_sent']:
            f.write("%s\n" % res)


    f.close()

