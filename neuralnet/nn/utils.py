__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'

import numpy as np
import logging


class Utils:
    logger = logging.getLogger(__name__)

    @staticmethod
    def softmax(x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    @staticmethod
    def save_model_parameters_theano(outfile, model):
        U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
        np.savez(outfile, U=U, V=V, W=W)
        Utils.logger.info("Saved model parameters to %s." % outfile)

    @staticmethod
    def load_model_parameters_theano(path, model):
        npz_file = np.load(path)
        U, V, W = npz_file["U"], npz_file["V"], npz_file["W"]
        model.hidden_dim = U.shape[0]
        model.word_dim = U.shape[1]
        model.U.set_value(U)
        model.V.set_value(V)
        model.W.set_value(W)
        Utils.logger.info("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))

    @staticmethod
    def load_csv_data(training_file, sentence_start_token, sentence_end_token, unknown_token, vocabulary_size):
        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        Utils.logger.debug("Reading CSV file...")
        import simplejson as json
        import nltk
        import itertools

        sentences = []
        with open(training_file, 'rb') as file:
            for line in file:
                post = json.loads(line)
                if "body" not in post:
                    continue
                body = post["body"]
                body_sentences = nltk.sent_tokenize(body.lower())
                for sentence in body_sentences:
                    sentences.append("%s %s %s" % (sentence_start_token, sentence, sentence_end_token))
        Utils.logger.info("Parsed %d sentences." % (len(sentences)))

        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        Utils.logger.info("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        Utils.logger.debug("Using vocabulary size %d." % vocabulary_size)
        Utils.logger.debug("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        Utils.logger.debug("Example sentence: '%s'" % sentences[0])
        Utils.logger.debug("Example sentence after Pre-processing: '%s'" % tokenized_sentences[0])

        return word_to_index, index_to_word, tokenized_sentences