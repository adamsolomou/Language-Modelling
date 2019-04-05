from sklearn.utils import shuffle
import collections
import numpy as np

BASE_VOCAB = {'<unk>': 0, '<bos>': 1, '<eos>': 2, '<pad>': 3}


class DataProcessing:
    vocab = None
    data_train = None
    data_validation = None

    def __init__(self, sentence_length, vocabulary_size):
        self.sentence_length = sentence_length
        self.vocabulary_size = vocabulary_size

    @property
    def train_corpus(self):
        return self.data_train

    @property
    def validation_corpus(self):
        return self.data_validation

    def preprocess_dataset(self, data_folder, train_file, validation_file):
        self.data_train = self.read_data(data_folder, train_file)
        self.data_validation = self.read_data(data_folder, validation_file)

    def read_data(self, data_folder, file_name):
        """
        Preprocessing step for evaluation and test set. Tokenize sentences and encode base on create vocabulary
        """
        tokenized_sentences = self.read_and_tokenize_sentences(data_folder, file_name)
        if self.vocab is None:
            self.construct_vocab(tokenized_sentences)
        return self.encode_text(tokenized_sentences)

    def read_and_tokenize_sentences(self, data_folder, file_name):
        """
        Reads corpus from specified location. Creates tokens per sentence by removing longer that allowed sentences.

        Parameters:
        -----------
        data_folder: string
            Folder location of corpus

        file_name: string
            File to read

        Returns
        -------
        tokenized_sentences: list
            list of list of tokens for sentences with at most self.sentence_length - 2 tokens
        """
        with open(data_folder + file_name, 'r') as f_data:
            tokenized_sentences = []
            for sentence in f_data.readlines():
                sentence_tokens = sentence.strip().split(" ")

                # Ignore sentences longer than sentence_len - 2
                if len(sentence_tokens) > self.sentence_length - 2:
                    continue

                tokenized_sentences.append(sentence_tokens)

            return tokenized_sentences

    def construct_vocab(self, tokenized_sentences, base_vocab=BASE_VOCAB):
        """
        Associate each word in the vocabulary with a (unique) ID. Construct the vocabulary dictionary where keys
        correspond to (unique) words in the vocabulary and the values correspond to the unique ID of the word.

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
            The vocabulary dictionary where keys correspond to (unique) words in the vocabulary and the values
            correspond to the unique ID of the word
        """
        counter = collections.Counter()

        for tokenized_sentence in tokenized_sentences:
            counter.update(tokenized_sentence)

        # Keep the vocab_size - (base_vocab size) most common words from the corpus
        most_common = counter.most_common(self.vocabulary_size - len(base_vocab))

        # Initialize the vocabulary
        vocab = dict(base_vocab)

        # Associate each word in the vocabulary with a unique ID number
        token_id = len(base_vocab)

        for token, _ in most_common:
            vocab[token] = token_id
            token_id += 1

        self.vocab = vocab

    def encode_text(self, tokenized_sentences):
        """
        Encode words in the text in terms of their ID in the vocabulary. Sentences that are longer than 30 tokens
        (including <bos> and <eos> are ignored).

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
        data = np.full(shape=(len(tokenized_sentences), self.sentence_length), fill_value=self.vocab['<pad>'],
                       dtype=int)

        data[:, 0] = self.vocab['<bos>']  # Beggining of sentence

        # Fill-in the rest of the data matrix
        for sentence_id, tokenized_sentence in enumerate(tokenized_sentences):
            position = 1
            for token in tokenized_sentence:
                if token in self.vocab:
                    data[sentence_id, position] = self.vocab[token]  # Within-vocabulary
                else:
                    data[sentence_id, position] = self.vocab['<unk>']  # Out-of-vocabulary

                position += 1

            data[sentence_id, position] = self.vocab['<eos>']  # End of sentence

        return data

    @staticmethod
    def get_batches(iterable, batch_size=64, do_shuffle=True):
        """
        Generate batches
        Parameters
        ----------
        iterable: list
            data to generate batches for

        batch_size: int

        do_shuffle: bool
            Whether to shuffle in each epoch
        """
        if do_shuffle:
            iterable = shuffle(iterable)

        length = len(iterable)
        for ndx in range(0, length, batch_size):
            iterable_batch = iterable[ndx: min(ndx + batch_size, length)]
            yield iterable_batch

