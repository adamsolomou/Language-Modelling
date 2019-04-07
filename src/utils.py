from gensim import models
import tensorflow as tf
import numpy as np


def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    """
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    """

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set


def train_batch(sentences_batch, lstm, train_step, global_step, session, summaries_merged, train_writer=None):
    """
    A single training step
    """
    feed_dict = {lstm.input_sentence: sentences_batch}

    fetches = [lstm.loss, summaries_merged, train_step, global_step]
    loss, summary, _, step = session.run(fetches, feed_dict)

    if train_writer is not None:
        train_writer.add_summary(summary, step)


def dev_step(sentences_batches, lstm, global_step, session, valid_writer=None, verbose=1):
    """
    Evaluate perplexity across validation dataset
    Parameters
    ----------
    sentences_batches: function returning list of batches
    lstm: LSTMCell
    global_step: tf.Variable(trainable=False)
    session: tf.Session()
    valid_writer: tensorboard writer
    verbose:
        if not None print perplexity scores
    """
    perplexities = []
    total_sentences = 0
    total_loss = 0
    total_perplexity = 0
    total_batches = 0

    # cannot feed all dataset into a single batch
    for sentences_batch in sentences_batches:
        fetches = [lstm.perplexity_per_sentence, lstm.loss, lstm.average_perplexity, global_step]
        perplexity_per_sentence_batch, lstm_loss, lstm_perplexity, step = \
            session.run(fetches, feed_dict={lstm.input_sentence: sentences_batch})

        perplexities = np.append(perplexities, perplexity_per_sentence_batch)
        total_sentences += len(sentences_batch)

        total_loss += lstm_loss
        total_perplexity += lstm_perplexity
        total_batches += 1

    current_step = tf.train.global_step(session, global_step)

    if valid_writer is not None:
        # get names of existing summaries to see results on the same graph
        loss_tag = lstm.summaries[0].name.split(':')[0]
        perplexity_tag = lstm.summaries[1].name.split(':')[0]

        summary_dev_loss = tf.Summary()
        summary_dev_loss.value.add(tag=loss_tag, simple_value=total_loss / total_batches)
        valid_writer.add_summary(summary_dev_loss, current_step)

        summary_dev_perplexity = tf.Summary()
        summary_dev_perplexity.value.add(tag=perplexity_tag, simple_value=total_perplexity / total_batches)
        valid_writer.add_summary(summary_dev_perplexity, current_step)

    if verbose is not None:
        print(f'Evaluation average perplexity per across sentences is {np.sum(perplexities) / total_sentences:.3f} '
              f'at step {current_step}')


def continue_sentence(sentence, session, lstm, inverse_vocab, eos_symbol, maximum_generated_length=20):
    """
    Continues the sentence provided according to the lstm model provided.

    Parameters
    ----------
    sentence: list
        list of integers corresponding to the token_ids
    session: tf.Session()
    lstm: LSTMCell
    inverse_vocab: dict
        token_ids to tokens mappings
    eos_symbol: int
        token_id of <eos>
    maximum_generated_length: int
        maximum sentence length to generate

    Returns
    -------
    The generated sentence
    """
    generated_sentence = list(map(inverse_vocab.get, sentence))[1:]

    # initial state for a single word
    states = np.zeros((lstm.state_size, 1, lstm.hidden_state_size))

    for word in sentence:
        feed_dict = {lstm.one_step_word_index: [word], lstm.one_step_state_1: states[0],
                     lstm.one_step_state_2: states[1]}
        states, next_word = session.run([lstm.one_step_new_state, lstm.one_step_next_word], feed_dict)

    # continue sentences
    words_generated = 0
    while True:
        # next_word has size (1,)
        # noinspection PyUnboundLocalVariable
        next_word = next_word[0]

        if next_word == eos_symbol:
            break

        generated_sentence.append(inverse_vocab[next_word])

        words_generated += 1
        if words_generated == maximum_generated_length:
            break

        feed_dict = {lstm.one_step_word_index: [next_word],
                     lstm.one_step_state_1: states[0], lstm.one_step_state_2: states[1]}
        states, next_word = session.run([lstm.one_step_new_state, lstm.one_step_next_word], feed_dict)

    return ' '.join(generated_sentence)
