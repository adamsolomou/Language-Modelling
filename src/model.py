import tensorflow as tf


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

    def __init__(self,
                 embedding_size,
                 hidden_state_size,
                 sentence_length,
                 vocabulary_size,
                 down_project_size=None,
                 pad_symbol=3):
        """
        Creates the LSTMCell

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
        self._hidden_state_size = hidden_state_size

        self._create_main_graph(embedding_size, hidden_state_size, sentence_length, vocabulary_size, down_project_size,
                                pad_symbol)
        self._create_one_step_graph(hidden_state_size, vocabulary_size, down_project_size)

        self._create_summaries()

    def _create_main_graph(self,
                           embedding_size,
                           hidden_state_size,
                           sentence_length,
                           vocabulary_size,
                           down_project_size,
                           pad_symbol):

        with tf.name_scope('lstm_input'):
            self._input_sentence = tf.placeholder(tf.int32, shape=[None, sentence_length], name='input_sentence')

        with tf.name_scope('embedding'):
            self._input_embeddings = weight_variable('input_embeddings', [vocabulary_size, embedding_size])
            sentence_embedding = tf.nn.embedding_lookup(self._input_embeddings, self._input_sentence,
                                                        name='sentence_embedding')

        with tf.name_scope("lstm"):
            self._lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_state_size, state_is_tuple=True,
                                                      initializer=tf.contrib.layers.xavier_initializer())

            # tuple of states for lstm
            states = self.get_initial_state(tf.shape(self._input_sentence)[0])

            outputs = []

            for time in range(sentence_length - 1):
                state_output, states = self._lstm_cell(sentence_embedding[:, time], states)

                output_t = vocabulary_projection_layer(state_output, vocabulary_size, down_project_size, (time > 0))

                outputs.append(output_t)

        with tf.name_scope('cross_entropy_loss'):
            """
            self.input_sentence[:, 1:] are the true predicted words avoid using 
            the pad predictions for the loss to avoid decreasing the model capacity
            
            mask all pad symbols for the calculation of the loss function
            """

            mask = tf.cast(tf.not_equal(self._input_sentence[:, 1:], pad_symbol), tf.float32)

            # transforms outputs to the required shape (batch_size, sentence_length - 1, vocabulary_size)
            total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.transpose(outputs, [1, 0, 2]),
                                                                        labels=self._input_sentence[:, 1:],
                                                                        name="cross_entropy") * mask

            # loss per sentence
            loss_per_batch_sample = tf.reduce_sum(total_loss, axis=1) / tf.reduce_sum(mask, axis=1)

            self._loss = tf.reduce_mean(loss_per_batch_sample)

        with tf.name_scope('perplexity'):
            # noinspection PyUnresolvedReferences
            self._perplexity_per_sentence = tf.exp(loss_per_batch_sample)
            self._average_perplexity = tf.reduce_mean(self._perplexity_per_sentence )

        with tf.name_scope("output_probabilities"):
            self._output_probabilities = tf.nn.softmax(outputs)

    def _create_one_step_graph(self,
                               hidden_state_size,
                               vocabulary_size,
                               down_project_size):
        """
        Creates a one step graph for the LSTM cell
        Given one word and the current states, calculates the next most probable word and the next states
        """

        with tf.name_scope("one_step"):
            self._one_step_word_index = tf.placeholder(tf.int32, [1], name="word_input")
            self._one_step_state_1 = tf.placeholder(tf.float32, [1, hidden_state_size], name="hidden_state_1")
            self._one_step_state_2 = tf.placeholder(tf.float32, [1, hidden_state_size], name="hidden_state_2")

            word_embedding = tf.nn.embedding_lookup(self._input_embeddings, self._one_step_word_index,
                                                    name='word_embedding')
            one_step_output, self._one_step_new_state = self._lstm_cell(word_embedding, (self._one_step_state_1,
                                                                                         self._one_step_state_2))
            # word with the maximum probability
            probabilities = vocabulary_projection_layer(one_step_output,
                                                        vocabulary_size, down_project_size, True)
            self._one_step_next_word = tf.argmax(probabilities, axis=1)

    def get_initial_state(self, size):
        """
        Returns zero state. No trainable option implemented hear
        """
        return self._lstm_cell.zero_state(size, tf.float32)

    def _create_summaries(self):
        """
        Creates summaries for tensorboard visualizations
        """
        self._loss_summary = tf.summary.scalar('loss', self._loss)
        self._perplexity_summary = tf.summary.scalar('average_perplexity_per_sentence', self._average_perplexity)

    @property
    def summaries(self):
        return [self._loss_summary, self._perplexity_summary]

    @property
    def input_sentence(self):
        return self._input_sentence

    @property
    def input_embeddings(self):
        return self._input_embeddings

    @property
    def loss(self):
        return self._loss

    @property
    def average_perplexity(self):
        return self._average_perplexity

    @property
    def perplexity_per_sentence(self):
        return self._perplexity_per_sentence

    @property
    def state_size(self):
        return 2

    @property
    def hidden_state_size(self):
        return self._hidden_state_size

    @property
    def one_step_word_index(self):
        return self._one_step_word_index

    @property
    def one_step_state_1(self):
        return self._one_step_state_1

    @property
    def one_step_state_2(self):
        return self._one_step_state_2

    @property
    def one_step_new_state(self):
        return self._one_step_new_state

    @property
    def one_step_next_word(self):
        return self._one_step_next_word
