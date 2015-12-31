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
        print("Reading CSV file...")
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
        print("Parsed %d sentences." % (len(sentences)))

        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique words tokens." % len(word_freq.items()))

        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        print("Using vocabulary size %d." % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        print("Example sentence: '%s'" % sentences[0])
        print("Example sentence after Pre-processing: '%s'" % tokenized_sentences[0])


        import numpy
        from neuralnet.nn.rnnnumpy import RNNNumpy
        numpy.random.seed(10)
        model = RNNNumpy(vocabulary_size)

        # Create the training data
        X_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        output_states, hidden_states = model.forward_propagation(X_train[10])
        print(output_states.shape)
        print(output_states)

        return ok
