import numpy as np
import tensorflow as tf


def train_batch(sentences_batch, lstm, train_step, global_step, session, train_writer, summaries_merged):
    """
    A single training step
    """
    feed_dict = {lstm.input_sentence: sentences_batch}

    fetches = [lstm.loss, summaries_merged, train_step, global_step]
    loss, summary, _, step = session.run(fetches, feed_dict)

    train_writer.add_summary(summary, step)


def dev_step(sentences_batches, lstm, global_step, session, valid_writer, verbose=1):
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

    # TODO hard coded tags to achieve average across batches results in same graph as train results
    summary_dev_loss = tf.Summary()
    summary_dev_loss.value.add(tag="cross_entropy_loss/loss", simple_value=total_loss / total_batches)
    valid_writer.add_summary(summary_dev_loss, current_step)

    summary_dev_perplexity = tf.Summary()
    summary_dev_perplexity.value.add(tag="perplexity/average_perplexity_per_sentence",
                                     simple_value=total_perplexity / total_batches)
    valid_writer.add_summary(summary_dev_perplexity, current_step)

    if verbose is not None:
        print(f'Evaluation average perplexity per across sentences is {np.sum(perplexities) / total_sentences:.3f} '
              f'at step {current_step}')


def continue_sentence(sentence, lstm):
    raise NotImplementedError


def load_embeddings():
    raise NotImplementedError
