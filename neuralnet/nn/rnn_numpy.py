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

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in numpy.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[numpy.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * numpy.sum(numpy.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = numpy.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N
    
    def bptt(self, x, y):
        T = len(y)

        # Perform forward propagation
        o, s = self.forward_propagation(x)

        # We accumulate the gradients in these variables
        dLdU = numpy.zeros(self.U.shape)
        dLdV = numpy.zeros(self.V.shape)
        dLdW = numpy.zeros(self.W.shape)
        delta_o = o
        delta_o[numpy.arange(len(y)), y] -= 1.

        # For each output backwards...
        for t in numpy.arange(T)[::-1]:
            dLdV += numpy.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in numpy.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += numpy.outer(delta_t, s[bptt_step-1])              
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

        return [dLdU, dLdV, dLdW]

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        import operator

        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, numpy.prod(parameter.shape)))
            
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = numpy.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]

                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = numpy.abs(backprop_gradient - estimated_gradient) / \
                    (numpy.abs(backprop_gradient) + numpy.abs(estimated_gradient))

                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    return
                it.iternext()

            print("Gradient check for parameter %s passed." % (pname))

