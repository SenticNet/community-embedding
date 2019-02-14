__author__ = 'ando'
import sklearn.mixture as mixture
import numpy as np
from utils.embedding import chunkize_serial

from scipy.stats import multivariate_normal
import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)


class Community2Vec(object):
    '''
    Class that train the community embedding
    '''
    def __init__(self, lr):
        self.lr = lr

    def fit(self, model, reg_covar=0, wc_prior=100, n_init=10):
        '''
        Fit the GMM model with the current node embedding and save the result in the model
        :param model: model injected to add the mixture parameters
        '''

        self.g_mixture = mixture.BayesianGaussianMixture(n_components=model.k,
                                                         reg_covar=reg_covar,
                                                         covariance_type='full',
                                                         n_init=n_init,
                                                         weight_concentration_prior=0.1,
                                                         weight_concentration_prior_type='dirichlet_process')

        log.info("Fitting: {} communities".format(model.k))
        self.g_mixture.fit(model.node_embedding)

        # diag_covars = []
        # for covar in g.covariances_:
        #     diag = np.diag(covar)
        #     diag_covars.append(diag)

        model.centroid = self.g_mixture.means_.astype(np.float32)
        model.covariance_mat = self.g_mixture.covariances_.astype(np.float32)
        model.inv_covariance_mat = self.g_mixture.precisions_.astype(np.float32)
        # model.pi = self.g_mixture.predict_proba(model.node_embedding).astype(np.float32)
        model.pi = self.g_mixture.weights_.astype(np.float32)

    def loss(self, nodes, model, beta, chunksize=150):
        """
        Forward function used to compute o3 loss
        :param input_labels: of the node present in the batch
        :param model: model containing all the shared data
        :param beta: trade off param
        """

        ret_loss = -1. * self.g_mixture.score_samples(model.node_embedding).sum()
        return ret_loss * (beta/model.k)

    def train(self, nodes, model, beta, chunksize=150, iter=1):
        for _ in range(iter):
            grad_input = np.zeros(model.node_embedding.shape).astype(np.float32)
            for node_index in chunkize_serial(map(lambda node: model.vocab[node].index, nodes), chunksize):
                input = model.node_embedding[node_index]
                batch_grad_input = np.zeros(input.shape).astype(np.float32)

                diff = np.expand_dims(input, axis=1) - np.expand_dims(model.centroid, axis=0)
                diff = np.transpose(diff, (1, 0, 2))
                for k, (d, inv_cov) in enumerate(zip(diff, model.inv_covariance_mat)):
                    batch_grad_input += model.pi[k] * np.sum(inv_cov * np.expand_dims(d, 1), 1)
                grad_input[node_index] += batch_grad_input

            grad_input *= (beta/model.k)

            model.node_embedding -= (grad_input.clip(min=-0.25, max=0.25)) * self.lr
