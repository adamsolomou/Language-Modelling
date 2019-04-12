from sklearn.utils import shuffle
import collections
import os

BASE_VOCAB = {'<unk>': 0, '<bos>': 1, '<eos>': 2, '<pad>': 3}


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


class DataProcessing:
    _vocab = None
    _inverse_vocab = None

    def __init__(self, sentence_length, vocabulary_size):
        self._sentence_length = sentence_length
        self._vocabulary_size = vocabulary_size

    def preprocess_dataset(self, data_folder, dataset_file, pad_to_sentence_length=True):
        return self._read_data(data_folder, dataset_file, pad_to_sentence_length=pad_to_sentence_length)

    def _read_data(self, data_folder, file_name, pad_to_sentence_length=True):
        """
        Preprocessing step for evaluation and test set. Tokenize sentences and encode base on create vocabulary
        """
        tokenized_sentences = self._read_and_tokenize_sentences(data_folder, file_name)
        if self._vocab is None:
            self._construct_vocab(tokenized_sentences)

        if pad_to_sentence_length:
            return self._encode_text(tokenized_sentences, padding_size=self._sentence_length)
        else:
            return self._encode_text(tokenized_sentences)

    def _read_and_tokenize_sentences(self, data_folder, file_name):
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
        with open(os.path.join(data_folder, file_name), 'r') as f_data:
            tokenized_sentences = []
            for sentence in f_data.readlines():
                sentence_tokens = sentence.strip().split(" ")

                # Ignore sentences longer than sentence_len - 2
                if len(sentence_tokens) > self._sentence_length - 2:
                    continue

                tokenized_sentences.append(sentence_tokens)

            return tokenized_sentences

    def _construct_vocab(self, tokenized_sentences, base_vocab=BASE_VOCAB):
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
        most_common = counter.most_common(self._vocabulary_size - len(base_vocab))

        # Initialize the vocabulary
        vocab = dict(base_vocab)

        # Associate each word in the vocabulary with a unique ID number
        token_id = len(base_vocab)

        for token, _ in most_common:
            vocab[token] = token_id
            token_id += 1

        # token to token_id mappings
        self._vocab = vocab
        # token_id to token mappings
        self._inverse_vocab = {v: k for k, v in vocab.items()}

    def _encode_text(self, tokenized_sentences, padding_size=None):
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
        encoded_sentences = []

        # Fill-in the data matrix
        for sentence_id, tokenized_sentence in enumerate(tokenized_sentences):
            encoded_sentence = [self._vocab['<bos>']]
            length = 1
            for token in tokenized_sentence:
                if token in self._vocab:
                    encoded_sentence.append(self._vocab[token])  # Within-vocabulary
                else:
                    encoded_sentence.append(self._vocab['<unk>'])  # Out-of-vocabulary

                length += 1

            if padding_size is not None:
                encoded_sentence.append(self._vocab['<eos>'])  # End of sentence
                encoded_sentence += [self._vocab['<pad>']] * (self._sentence_length - length - 1)
            encoded_sentences.append(encoded_sentence)

        return encoded_sentences

    def decode_sentence(self, sentence):
        """
        Parameters
        ----------
        sentence: list
            List of token_ids, first token is <bos> and is ignored

        Returns
        -------
        Generates an sentence omitting starting and ending symbols
        """
        return ' '.join(list(map(self._inverse_vocab.get, sentence))[1:])

    @property
    def vocab(self):
        return self._vocab
