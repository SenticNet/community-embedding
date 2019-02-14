
import sklearn.mixture as mixture
import numpy as np
from os import path
import plot_utils

def optimise(X, pi, mu, inv_covariance_mat, lr, iter):
    for i in range(iter):

        grad_input = np.zeros(X.shape).astype(np.float32)
        for node_index in [list(range(15)), list(range(15, 34))]:
            input = X[node_index]
            batch_grad_input = np.zeros(input.shape).astype(np.float32)

            diff = np.expand_dims(input, axis=1) - np.expand_dims(mu, axis=0)
            diff = np.transpose(diff, (1, 0, 2))
            for k, (d, inv_cov) in enumerate(zip(diff, inv_covariance_mat)):
                batch_grad_input += pi[k] * np.sum(inv_cov * np.expand_dims(d, 1), 1)
            grad_input[node_index] += batch_grad_input

        grad_input *= lr

        X -= grad_input.clip(min=-0.25, max=0.25)

    return X


def load_embedding(file_name, path_='data', ext=".txt"):
    """
    Load the embedding saved in a .txt file
    :param file_name: name of the file to load
    :param path: location of the file
    :param ext: extension of the file to load
    :return:
    """
    ret = []
    with open(path.join(path_, file_name + ext), 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            node_values = [float(val) for val in tokens[1].strip().split(' ')]
            ret.append(node_values)
    ret = np.array(ret, dtype=np.float32)
    return ret

embedding = load_embedding("karate_deepwalk_neg-4_ws-3_alpha-0.1", path_="data/Karate")



g_mixture = mixture.GaussianMixture(n_components=2,
                                reg_covar=0.001,
                                covariance_type='full',
                                n_init=30)

g_mixture.fit(embedding)


centroid = g_mixture.means_.astype(np.float32)
covariance_mat = g_mixture.covariances_.astype(np.float32)
inv_covariance_mat = g_mixture.precisions_.astype(np.float32)
pi = g_mixture.weights_.astype(np.float32)


label = g_mixture.predict(embedding)

print("prob:\n{}".format(g_mixture.predict_proba(embedding).astype(np.float32)))

print("loss: {}".format(-1. * g_mixture.score_samples(embedding).sum()))


avg_dist = np.expand_dims(embedding, axis=1) - np.expand_dims(centroid, axis=0)
avg_dist = np.mean(np.linalg.norm(np.array([dff[idx] for dff, idx in zip(avg_dist, label)]), axis=1))
print("avg_dist to center: {}".format(avg_dist))

for i in range(50):
    color = (np.arange(34) / 34).tolist()
    plot_utils.node_space_plot_2D_elipsoid(embedding,
                                           centroid,
                                           covariance_mat,
                                           color, str(i))

    embedding = optimise(embedding, pi, centroid, inv_covariance_mat, 0.001, 1)

    print("loss: {}/{}".format(-1. * g_mixture.score_samples(embedding).sum(), i))

    avg_dist = np.expand_dims(embedding, axis=1) - np.expand_dims(centroid, axis=0)
    avg_dist = np.mean(np.linalg.norm(np.array([dff[idx] for dff, idx in zip(avg_dist, label)]), axis=1))
    print("avg_dist to center: {}".format(avg_dist))
