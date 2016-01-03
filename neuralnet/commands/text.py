__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'

import logging
import os
import numpy
from .base import ICommand
from zope.interface import implementer
from neuralnet.nn.utils import Utils

logger = logging.getLogger(__name__)

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_TRAINING_FILE = 'data/RC_2010-02-10k.json'


@implementer(ICommand)
class TextCommand:
    command = "text"
    usage = "text <foo>"
    short = "Run some RNN text generation."
    docs = """See the nn.yaml configuration for details."""

    def __init__(self):
        self.vocabulary_size = _VOCABULARY_SIZE
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

    def setenv(self, env):
        self.env = env

    def validate(self, args):
        if len(args) != 1:
            raise RuntimeError("Number of arguments must be exactly 1")
        #check_stage(self.env, args[0])
        return args

    def execute(self, stage):
        ok = True

        (word_to_index, index_to_word, tokenized_sentences) = Utils.load_csv_data(
            _TRAINING_FILE, self.sentence_start_token, self.sentence_end_token,
            self.unknown_token, self.vocabulary_size)

        self.rnn_theano(word_to_index, index_to_word, tokenized_sentences)
        return ok

    def rnn_theano(self, word_to_index, index_to_word, tokenized_sentences):
        # Create the training data
        x_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        model = self.get_theano_model(x_train, y_train)
        model.sgd_step(x_train[10], y_train[10], 0.005)

        num_sentences = 10
        sentence_min_length = 7

        for i in range(num_sentences):
            sent = []
            # We want long sentences, not sentences with one or two words
            while len(sent) < sentence_min_length:
                sent = self.generate_sentence(model, word_to_index, index_to_word)
            print(" ".join(sent))

    def generate_sentence(self, model, word_to_index, index_to_word):
        # We start the sentence with the start token
        new_sentence = [word_to_index[self.sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[self.sentence_end_token]:
            #next_word_probs = model.forward_propagation(new_sentence)
            next_word_probs = model.predict(new_sentence)
            sampled_word = word_to_index[self.unknown_token]

            # We don't want to sample unknown words
            while sampled_word == word_to_index[self.unknown_token]:
                samples = numpy.random.multinomial(1, next_word_probs[-1])
                sampled_word = numpy.argmax(samples)
                #logger.debug("sampled_word: %s" % index_to_word[sampled_word])

            new_sentence.append(sampled_word)

        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str

    def get_theano_model(self, x_train, y_train):
        from neuralnet.nn.rnn_theano import RNNTheano

        trained_model_data = './data/trained-model-theano.npz'

        model = RNNTheano(self.vocabulary_size, hidden_dim=50)
        if os.path.isfile(trained_model_data):
            logger.info("found trained model, loading instead of training new: %s" % trained_model_data)
            Utils.load_model_parameters_theano(trained_model_data, model)
        else:
            logger.info("training new model...")
            numpy.random.seed(10)

            self.train_with_sgd(model, x_train, y_train, nepoch=5)

            logger.info("saving trained model to: %s" % trained_model_data)
            Utils.save_model_parameters_theano(trained_model_data, model)

        return model

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(self, model, x_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        from datetime import datetime
        from time import time

        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        total_steps_to_train = len(y_train)*nepoch
        total_time = 0

        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if epoch > 0 and epoch % evaluate_loss_after == 0:
                loss = model.calculate_loss(x_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate *= 0.5
                    logger.info("Setting learning rate to %f" % learning_rate)

            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                before = time()
                model.sgd_step(x_train[i], y_train[i], learning_rate)
                after = time()

                num_examples_seen += 1
                total_time += (after - before)*1000
                average_step_time = total_time/num_examples_seen
                self.log_time(i, epoch, nepoch, average_step_time, total_steps_to_train, num_examples_seen)

    def log_time(self, i, epoch, nepoch, average_step_time, total_steps_to_train, num_examples_seen):
        if i % 100 != 0:
            return

        from math import floor
        avg_step_time = round(average_step_time, 2)
        est_total_time = average_step_time*total_steps_to_train
        est_elapsed_time = average_step_time*num_examples_seen
        time_left = floor((est_total_time - est_elapsed_time) / 1000)

        formatted_time = "%ss" % time_left
        if time_left > 3600:
            seconds = time_left % 60
            minutes = ((time_left - seconds)/60) % 60
            hours = (time_left - minutes*60 - seconds) / 3600
            formatted_time = "%s hours, %s minutes, %s seconds" % (int(hours), int(minutes), int(seconds))
        elif time_left > 60:
            seconds = time_left % 60
            minutes = (time_left - seconds)/60
            formatted_time = "%s minutes, %s seconds" % (int(minutes), int(seconds))

        logger.debug("epoch %s/%s, examples %s/%s, avg step time %sms, time left %s" %
                     (epoch, nepoch, num_examples_seen, total_steps_to_train,
                     avg_step_time, formatted_time))

    def rnn_numpy(self, vocabulary_size, word_to_index, tokenized_sentences):
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
