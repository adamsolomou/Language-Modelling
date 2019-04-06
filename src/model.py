import tensorflow as tf
import math


def weight_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    """
    Creates a variable with a specified initializer
    """
    variable = tf.get_variable(name, initializer=initializer(shape))
    return variable


def vocabulary_projection_layer(input_, output_dimension, down_project_size, reuse):
    with tf.variable_scope("output_layer", reuse=reuse):
        if down_project_size is not None:
            input_ = tf.layers.dense(input_, down_project_size, use_bias=True, name='down_projection_projection')
        return tf.layers.dense(input_, output_dimension, use_bias=True, name='vocabulary_size_projection')


class LSTMCell:
    """
    A LSTM implementation in TensorFlow.
    """
    # TODO add properties
    def __init__(self, embedding_size, hidden_state_size, sentence_length,
                 vocabulary_size, down_project_size=None, pad_symbol=3):
        """
        Creates the lstm graph

        Parameters
        ----------
        embedding_size: int
        hidden_state_size: int
        sentence_length: int
        vocabulary_size: int
        down_project_size: int
            size of down-projected hidden state before the application of the softmax
        pad_symbol: int
            token_id for the <pad> symbol
        """

        with tf.name_scope('lstm_input'):
            self.input_sentence = tf.placeholder(tf.int32, shape=[None, sentence_length], name='input_sentence')

        with tf.name_scope('embedding'):
            self.input_embeddings = weight_variable('input_embeddings', [vocabulary_size, embedding_size])
            sentence_embedding = tf.nn.embedding_lookup(self.input_embeddings, self.input_sentence,
                                                        name='sentence_embedding')

        with tf.variable_scope("lstm"):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_state_size, state_is_tuple=True,
                                                initializer=tf.contrib.layers.xavier_initializer())

            # tuple of states for lstm
            states = lstm_cell.zero_state(tf.shape(self.input_sentence)[0], tf.float32)

            outputs = []

            for time in range(sentence_length - 1):
                state_output, states = lstm_cell(sentence_embedding[:, time], states)

                output_t = vocabulary_projection_layer(state_output, vocabulary_size, down_project_size, (time > 0))

                outputs.append(output_t)

        with tf.name_scope('cross_entropy_loss'):
            """
            self.input_sentence[:, 1:] are the true predicted words avoid using 
            the pad predictions for the loss to avoid decreasing the model capacity
            """

            mask = tf.cast(tf.not_equal(self.input_sentence[:, 1:], pad_symbol), tf.float32)

            # transforms outputs to the required shape (batch_size, sentence_length - 1, vocabulary_size)
            total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.transpose(outputs, [1, 0, 2]),
                                                                        labels=self.input_sentence[:, 1:],
                                                                        name="cross_entropy") * mask

            # loss per sentence
            loss_per_batch_sample = tf.reduce_sum(total_loss, axis=1) / tf.reduce_sum(mask, axis=1)

            self.loss = tf.reduce_mean(loss_per_batch_sample)

            self.loss_summary = tf.summary.scalar('loss', self.loss)

        with tf.name_scope('perplexity'):
            # multiplying with math.log(2) makes cross entropy with a base of 2
            self.perplexity_per_sentence = tf.math.pow(tf.constant(2, dtype=tf.float32),
                                                       loss_per_batch_sample / math.log(2))
            self.average_perplexity = tf.reduce_mean(self.perplexity_per_sentence)

            self.perplexity_summary = tf.summary.scalar('average_perplexity_per_sentence', self.average_perplexity)

        with tf.name_scope("output_probabilities"):
            self.output_probabilities = tf.nn.softmax(outputs)

    @property
    def summaries(self):
        return [self.loss_summary, self.perplexity_summary]
