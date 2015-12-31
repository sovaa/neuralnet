__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'

import numpy


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Randomly initialize the network parameters
        self.U = numpy.random.uniform(-numpy.sqrt(1./word_dim), numpy.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        time_steps = len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        hidden_states = numpy.zeros((time_steps + 1, self.hidden_dim))
        hidden_states[-1] = numpy.zeros(self.hidden_dim)

        # The outputs at each time step. Again, we save them for later.
        output_states = numpy.zeros((time_steps, self.word_dim))

        # For each time step...
        for time_step in numpy.arange(time_steps):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            hidden_states[time_step] = numpy.tanh(self.U[:, x[time_step]] + self.W.dot(hidden_states[time_step-1]))
            output_states[time_step] = self.softmax(self.V.dot(hidden_states[time_step]))
        return [output_states, hidden_states]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        output_states, hidden_states = self.forward_propagation(x)
        return numpy.argmax(output_states, axis=1)

    def softmax(self, w_in, t = 1.0):
        e = numpy.exp(numpy.array(w_in) / t)
        dist = e / numpy.sum(e)
        return dist

