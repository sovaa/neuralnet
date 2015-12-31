__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'

import logging
import os
from .base import ICommand
from zope.interface import implementer

logger = logging.getLogger(__name__)


_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))


@implementer(ICommand)
class TextCommand:
    command = "text"
    usage = "text <foo>"
    short = "Run some RNN text generation."
    docs = """See the nn.yaml configuration for details."""

    def setenv(self, env):
        self.env = env

    def validate(self, args):
        if len(args) != 1:
            raise RuntimeError("Number of arguments must be exactly 1")
        #check_stage(self.env, args[0])
        return args

    def execute(self, stage):
        ok = True

        vocabulary_size = _VOCABULARY_SIZE
        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"

        # Read the data and append SENTENCE_START and SENTENCE_END tokens
        logger.debug("Reading CSV file...")
        import simplejson as json
        import nltk
        import itertools

        sentences = []
        with open('data/RC_2010-02-small.json', 'rb') as file:
            for line in file:
                post = json.loads(line)
                if "body" not in post:
                    continue
                body = post["body"]
                sentence = nltk.sent_tokenize(body.lower())
                sentences.append("%s %s %s" % (sentence_start_token, sentence, sentence_end_token))
        logger.info("Parsed %d sentences." % (len(sentences)))

        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        logger.info("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        logger.debug("Using vocabulary size %d." % vocabulary_size)
        logger.debug("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        logger.debug("Example sentence: '%s'" % sentences[0])
        logger.debug("Example sentence after Pre-processing: '%s'" % tokenized_sentences[0])

        self.rnn_theano(vocabulary_size, word_to_index, tokenized_sentences)

        return ok

    def rnn_theano(self, vocabulary_size, word_to_index, tokenized_sentences):
        import numpy

        numpy.random.seed(10)

        # Create the training data
        x_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        model = self.get_theano_model(vocabulary_size, x_train, y_train)
        model.sgd_step(x_train[10], y_train[10], 0.005)

    def get_theano_model(self, vocabulary_size, x_train, y_train):
        from neuralnet.nn.utils import Utils
        from neuralnet.nn.rnn_theano import RNNTheano
        import os.path
        trained_model_data = './data/trained-model-theano.npz'

        model = RNNTheano(vocabulary_size, hidden_dim=50)
        if os.path.isfile(trained_model_data):
            logger.info("found trained model, loading instead of training new: %s" % trained_model_data)
            Utils.load_model_parameters_theano(trained_model_data, model)
        else:
            logger.info("training new model...")
            self.train_with_sgd(model, x_train, y_train, nepoch=5)
            logger.info("saving trained model to: %s" % trained_model_data)
            Utils.save_model_parameters_theano(trained_model_data, model)

        return model

    def rnn_numpy(self, vocabulary_size, word_to_index, tokenized_sentences):
        import numpy
        from neuralnet.nn.rnn_numpy import RNNNumpy

        numpy.random.seed(10)
        model = RNNNumpy(vocabulary_size, 100, bptt_truncate=1000)

        # Create the training data
        X_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        output_states, hidden_states = model.forward_propagation(X_train[10])
        logger.debug(output_states.shape)
        logger.debug(output_states)

        # Limit to 1000 examples to save time
        logger.debug("Expected Loss for random predictions: %f" % numpy.log(vocabulary_size))
        logger.debug("Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000]))

        grad_check_vocab_size = 100
        numpy.random.seed(10)
        model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
        model.gradient_check([0, 1, 2, 3], [1, 2, 3, 4])

        numpy.random.seed(10)
        # Train on a small subset of the data to see what happens
        model = RNNNumpy(vocabulary_size)
        self.train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, model, x_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        from datetime import datetime

        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0

        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:
                loss = model.calculate_loss(x_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    logger.info("Setting learning rate to %f" % learning_rate)
                #sys.stdout.flush()

            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sgd_step(x_train[i], y_train[i], learning_rate)
                num_examples_seen += 1