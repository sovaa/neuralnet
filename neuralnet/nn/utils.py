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
        npzfile = np.load(path)
        U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
        model.hidden_dim = U.shape[0]
        model.word_dim = U.shape[1]
        model.U.set_value(U)
        model.V.set_value(V)
        model.W.set_value(W)
        Utils.logger.info("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))