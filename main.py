__author__ = 'ando'
import os
import random
from multiprocessing import cpu_count
import logging as log


import numpy as np
import psutil
from math import floor
from model_src.model import Model
from model_src.context_embeddings import Context2Vec
from model_src.node_embeddings import Node2Vec
from model_src.community_embeddings import Community2Vec
import utils.IO_utils as io_utils
import utils.graph_utils as graph_utils
import timeit

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)




p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

if __name__ == "__main__":

    #Reading the input parameters form the configuration files
    number_walks = 10                       # number of walks for each node
    walk_length = 80                        # length of each walk
    representation_size = [128] # size of the embedding
    num_workers = 10                        # number of thread
    num_iter = 1                            # number of overall iteration
    reg_covar = 0.0001                     # regularization coefficient to ensure positive covar
    input_file = 'Dblp'                # name of the input file
    output_file = 'Dblp'               # name of the output file
    batch_size = 50
    window_size = 10    # windows size used to compute the context embedding
    negative = 5        # number of negative sample
    lr = 0.025            # learning rate
    down_sampling = 0.
    
    alpha = 0.1
    beta = 1.
    k = 20




    walks_filebase = os.path.join('data', output_file)            # where read/write the sampled path



    #CONSTRUCT THE GRAPH
    G = graph_utils.load_matfile(os.path.join('./data', input_file, input_file + '.mat'), undirected=True)
    # Sampling the random walks for context
    for d in representation_size:
        log.info("\n-------------- d:{} --------------".format(d))
        log.info("sampling the paths")

        walk_files = ["data/{}/{}.walks.0{}".format(input_file, input_file, i) for i in range(10)]
        vertex_counts = graph_utils.count_textfiles(walk_files, num_workers)
        model = Model(vertex_counts,
                      size=d,
                      down_sampling=down_sampling,
                      table_size=100000000,
                      # table_size=100000,
                      input_file=os.path.join(input_file, input_file),
                      path_labels="./data")


        #Learning algorithm
        node_learner = Node2Vec(workers=num_workers, negative=negative, lr=lr)
        cont_learner = Context2Vec(window_size=window_size, workers=num_workers, negative=negative, lr=lr)
        com_learner = Community2Vec(lr=lr)


        context_total_path = G.number_of_nodes() * number_walks * walk_length
        edges = np.array(G.edges())
        log.debug("context_total_path: %d" % (context_total_path))
        log.debug('node total edges: %d' % G.number_of_edges())

        log.info('\n_______________________________________')
        log.info('\t\tPRE-TRAINING\n')
        ###########################
        #   PRE-TRAINING          #
        ###########################
        model.reset_communities_weights(k)

        node_learner.train(model,
                           edges=edges,
                           iter=1,
                           chunksize=batch_size)

        cont_learner.train(model,
                           paths=graph_utils.combine_files_iter(walk_files),
                           total_nodes=context_total_path,
                           alpha=1,
                           chunksize=batch_size)

        model.save("{}_pre-training".format(output_file))

        ###########################
        #   EMBEDDING LEARNING    #
        ###########################
        iter_node = floor(context_total_path/G.number_of_edges())
        iter_com = floor(context_total_path/G.number_of_edges())
        for it in range(num_iter):
            log.info('\n_______________________________________\n')
            log.info('\t\tITER-{}\n'.format(it))
            log.info('using alpha:{}\tbeta:{}\titer_com:{}\titer_node: {}'.format(alpha, beta, iter_com, iter_node))
            start_time = timeit.default_timer()

            com_learner.fit(model, reg_covar=reg_covar, n_init=10)
            node_learner.train(model,
                               edges=edges,
                               iter=iter_node,
                               chunksize=batch_size)


            com_learner.train(G.nodes(), model, beta, chunksize=batch_size, iter=iter_com)

            cont_learner.train(model,
                               paths=graph_utils.combine_files_iter(walk_files),
                               total_nodes=context_total_path,
                               alpha=alpha,
                               chunksize=batch_size)

            log.info('time: %.2fs' % (timeit.default_timer() - start_time))

            io_utils.save_embedding(model.node_embedding, model.vocab,
                                    file_name="{}_alpha-{}_beta-{}_ws-{}_neg-{}_lr-{}_icom-{}_ind-{}_k-{}_d-{}".format(output_file,
                                                                                                                   alpha,
                                                                                                                   beta,
                                                                                                                   window_size,
                                                                                                                   negative,
                                                                                                                   lr,
                                                                                                                   iter_com,
                                                                                                                   iter_node,
                                                                                                                    model.k,
                                                                                                                    d))

