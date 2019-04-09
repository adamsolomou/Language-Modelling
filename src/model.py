import sklearn 
import numpy as np 
import tensorflow as tf

from tqdm import tqdm

class LanguageModel(object):
    
    def __init__(self, vocab, sentence_len=30, embedding_dim=100, hidden_size=512):

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.sentence_len = sentence_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Construct the computational graph 
        self._model_graph()

    def _model_graph(self):
        """
        Construct the computational graph of the RNN

        Returns:
        --------
        self : object
            An instance of self
        """

        with tf.name_scope("Model"):
            # Input sentence
            self._x = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len], name="input_sentences")

            # Embedding matrix 
            self.embeddings = tf.get_variable(name="embeddings", shape=[self.vocab_size, self.embedding_dim], 
                                dtype=tf.float32, initializer=tf.initializers.random_uniform(-0.25, 0.25))

            # Embedding representation of input sentence
            self.x = tf.nn.embedding_lookup(self.embeddings, self._x) # [BATCH_SIZE, SENTENCE_LEN, EMBEDDING_DIM]

            # LSTM cell
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, 
                                    initializer=tf.contrib.layers.xavier_initializer(), 
                                    name="lstm_cell")

            # Initial state 
            state = cell.zero_state(batch_size=tf.shape(self._x)[0], dtype=tf.float32)

            # Initialize parameters 
            total_loss = []
            self.y_pred_prob = []

            with tf.variable_scope("RNN"):
                # RNN loop 
                for t in range(self.sentence_len - 1):
                    # Reuse RNN variables 
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()  
                        assert tf.get_variable_scope().reuse==True

                    # Batch of words [BATCH_SIZE, EMBEDDING_DIM]
                    word = self.x[:,t,:]

                    # Labels [BATCH_SIZE]
                    labels = self._x[:,t+1]

                    # Cell call 
                    output, state = cell(word, state)

                    # Compute logits
                    logits = tf.layers.dense(inputs=output, units=self.vocab_size, activation=None, 
                                kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=(t > 0), name="output_layer")

                    # Append batch loss per word 
                    total_loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

                    # Append softmax activations per word
                    self.y_pred_prob.append(tf.nn.softmax(logits, name="softmax_activation"))

            with tf.name_scope("Loss_Computation"):
                # Transpose loss tensor to bring it to shape [BATCH_SIZE, SENTENCE_LEN-1]
                total_loss = tf.transpose(total_loss)

                # Ignore '<pad>' symbols in loss computation 
                mask = tf.cast(tf.math.not_equal(self._x[:,1:], self.vocab['<pad>']), dtype=tf.float32)

                total_loss = mask * total_loss

                # Add up losses in each sentence and divide by sentence len (excluding '<pad>' symbols)
                loss_per_sentence = tf.reduce_sum(total_loss, axis=1) / tf.reduce_sum(mask, axis=1)

                # Total batch loss
                self.loss = tf.reduce_mean(loss_per_sentence)

            with tf.name_scope("Perplexity_Computation"):

                self.perplexity_per_sentence = tf.exp(loss_per_sentence)

                self.mean_perplexity = tf.reduce_mean(self.perplexity_per_sentence)

            return self

    def optimizer(self, clip_norm=5.0):
        """
        Compute, clip and gradients 

        Parameters:
        -----------
        clip_norm : A 0-D (scalar) Tensor > 0
            The clipping ratio

        Returns:
        --------
        self : object 
            An instance of self
        """

        with tf.name_scope("Gradient_Computation"):
            self.optimizer = tf.train.AdamOptimizer()

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params, name="compute_gradients")

            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0, name="clip_gradients")

        with tf.name_scope("Apply_Gradients"):
            self.train_step = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_params))

        return self

    def fit(self, x, val_data, epochs=1, batch_size=64, shuffle=True):
        """
        Trains the model for a given number of epochs (iterations on a dataset).

        Parameters:
        -----------
        x : array-like, shape=(n_sentences, sentence_len)
            Training sentences encoded in terms of IDs in the vocabulary 

        batch_size : int
            Number of samples per gradient update. If unspecified, batch_size will default to 64.

        epochs : int 
            Number of epochs to train the model. An epoch is an iteration over the entire x data provided.

        val_data : array-like, shape=(n_sentences, sentence_len)
            Data on which to evaluate the loss and perplexity at the end of each epoch. The model will not be trained on this data. 
        
        shuffle : Boolean
            Whether to shuffle the training data before each epoch

        Returns:
        --------
        self : object 
            An instance of self
        """

        # Define Optimizer
        self.optimizer()

        # Saver
        saver = tf.train.Saver()

        # Initialize the variables 
        init = tf.global_variables_initializer()

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            # Number of sentences
            N_train = x.shape[0]
            N_valid = val_data.shape[0]

            # Floor division 
            train_steps = N_train // batch_size
            valid_steps = N_valid // batch_size

            display_step = 3000

            for epoch in range(epochs):

                print('Epoch %i' %epoch)

                if shuffle: 
                    x = sklearn.utils.shuffle(x)

                e_train_loss = 0

                for i in tqdm(range(train_steps)):

                    # Get mini-batch 
                    batch_x = x[i*batch_size:(i+1)*batch_size, :]

                    print('check 1')

                    # Training step 
                    sess.run(self.train_step, feed_dict={self._x: batch_x}, options=run_options)

                    print('check 2')

                    # Accumulate training loss
                    e_train_loss += sess.run(self.loss, feed_dict={self._x: batch_x})

                    # Report validation perplexity on the validation set 
                    if i % display_step == 0:
                        # Initialize perplexity tensor
                        perplexity_per_sentence = [] 

                        print('check 3')

                        # Evaluate on validation data 
                        for j in range(valid_steps):

                            batch_x = val_data[j*batch_size:(j+1)*batch_size, :]

                            # Append perplexity per sentence
                            perplexity_per_sentence = tf.concat([perplexity_per_sentence,
                                                                sess.run(self.perplexity_per_sentence, 
                                                                        feed_dict={self._x: batch_x})], axis=0)

                        # Add loss and perplexity for the remaining validation sentences
                        batch_x = x[valid_steps*batch_size:, :]

                        perplexity_per_sentence = tf.concat([perplexity_per_sentence,
                                                            sess.run(self.perplexity_per_sentence, feed_dict={self._x: batch_x})], axis=0)

                        print('Average perplexity over validation sentences at step %i: %f' %(i, np.mean(perplexity_per_sentence)))

                print('Average training loss per step at epoch %i: %f' %(epoch, e_train_loss/train_steps))

                perplexity_per_sentence = []

                # Evaluate on validation data at the end of each epoch 
                for j in range(valid_steps):

                    batch_x = val_data[j*batch_size:(j+1)*batch_size, :]

                    # Append perplexity per sentence
                    perplexity_per_sentence = tf.concat([perplexity_per_sentence,
                                                        sess.run(self.perplexity_per_sentence, 
                                                                feed_dict={self._x: batch_x})], axis=0)

                # Add loss and perplexity for the remaining validation sentences
                batch_x = val_data[valid_steps*batch_size:, :]

                # Append perplexity per sentence
                perplexity_per_sentence = tf.concat([perplexity_per_sentence,
                                                    sess.run(self.perplexity_per_sentence, 
                                                            feed_dict={self._x: batch_x})], axis=0)


                print('Mean perplexity over validation sentences at epoch %i: %f' %(epoch, np.mean(perplexity_per_sentence)))

            # TO BE FIXED SO THAT WE SAVE THE BEST MODEL!
            save_path = saver.save(sess, "model.ckpt")

            print("Model saved in path: %s" % save_path)

        return self

    def evaluate(self, x, batch_size=64):
        """
        Compute loss and perplexity values for the model in test mode. 

        Parameters:
        -----------
        x : array-like, shape=(n_sentences, sentence_len)
            Training sentences encoded in terms of IDs in the vocabulary. 

        batch_size : int
            Number of samples per gradient update. If unspecified, batch_size will default to 64.

        Returns:
        --------
        A dictionary with accumulated loss, mean_perplexity and perplexity per_sentence.   
        """

        N = x.shape[0]

        # Floor division
        n_steps = N // batch_size

        loss = 0
        perplexity_per_sentence = [] 

        # Saver
        saver = tf.train.Saver()

        with tf.Session() as sess:

            # Restore trained model 
            saver.restore(sess, "model.ckpt")
            print("Model restored.")

            for i in range(n_steps):

                # Get mini-batch 
                batch_x = x[i*batch_size:(i+1)*batch_size, :]

                # Accumulate loss
                loss += sess.run(self.loss, feed_dict={self._x: batch_x})

                # Append perplexity per sentence
                perplexity_per_sentence = tf.concat([perplexity_per_sentence,
                                                    sess.run(self.perplexity_per_sentence, feed_dict={self._x: batch_x})], axis=0)


            # Add loss and perplexity for the remaining test sentences
            batch_x = x[n_steps*batch_size:, :]

            loss += sess.run(self.loss, feed_dict={self._x: batch_x})

            perplexity_per_sentence = tf.concat([perplexity_per_sentence,
                                                    sess.run(self.perplexity_per_sentence, feed_dict={self._x: batch_x})], axis=0)

            # Compute average from all mini-batches
            mean_perplexity = np.mean(perplexity_per_sentence)

        return {'loss': loss, 'mean_perp': mean_perplexity, 'perp_per_sent': perplexity_per_sentence}

    def total_params(self):
        # Compute the total number of parameters
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

