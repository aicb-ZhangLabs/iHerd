# -*- coding: UTF-8 -*
import os
import time
import tensorflow as tf
from coarsen import create_coarse_graph
from utils import normalized, graph_to_adj
from numpy import *
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

import json
import pickle

def print_coarsen_info(ctrl, g):
    cnt = 0
    while g is not None:
        ctrl.logger.info("Level " + str(cnt) + " --- # nodes: " + str(g.node_num))
        g = g.coarser
        cnt += 1


def multilevel_embed(ctrl, graph, match_method, basic_embed, refine_model, args): #, AttrMat):
    '''This method defines the multilevel embedding method.'''

    start = time.time()
    # Step-1: Graph Coarsening.

    original_graph = graph
    stru_comms = {}
    comms={}
    coarsen_level = ctrl.coarsen_level

    if ctrl.refine_model.double_base:  # if it is double-base, it will need to do one more layer of coarsening
       coarsen_level += 1
    print("coarsen_level:" + str(coarsen_level ))
    for i in range(coarsen_level):
        # Take out the part of attributes
        stru_comms[i] = match_method(graph)

        coarse_graph, comms[i] = create_coarse_graph(stru_comms=stru_comms[i], graph=graph)

        # Specifically set up for "grarep" to solve the problem that node_num smaller than dimensions for each step.
        # the default parameter Kstep in 'grarep' == 4
        # if ctrl.basic_embed == 'grarep':
        #     if coarse_graph.node_num < int(ctrl.embed_dim / 8):
        #         print(f"coraser graph has too few nodes for 'grarep', use a finer graph at level {i} instead, which has {graph.node_num} nodes.")
        #         break

        graph = coarse_graph

    struct_path = args.data_folder+"Output/Structure/"+args.subdata+".pkl"
    if not os.path.exists(struct_path):
           # Create a new directory because it does not exist
           os.makedirs(struct_path)

    with open(struct_path, 'wb') as fp:
        pickle.dump(stru_comms, fp)

    if ctrl.debug_mode and graph.node_num < 1e3:
        assert np.allclose(graph_to_adj(graph).A, graph.A.A), "Coarser graph is not consistent with Adj matrix"
    print_coarsen_info(ctrl, original_graph)
    print("coarse ok")

    # Step-2 : Base Embedding
    if ctrl.refine_model.double_base:
        graph = graph.finer
    embedding = basic_embed(ctrl, graph)
    embedding = normalized(embedding, per_feature=False)
    #only for Single-granularity Structure-only Network Embedding

    #embedding=concatenate((0.5*embedding, graph.Attr),axis=1)

    #print("n" + str(n))
    # dataMat = np.ones((n, args.embed_dim))
    # #print("dataMat shape" + str(dataMat.shape))
    # dataMat = normalized(dataMat, per_feature=False)  # for Flickr, BlogCatalog
    # #print("dataMat shape normalized" + str(dataMat.shape))
    # pca = PCA(n_components=min(args.embed_dim, n))
    # pca.fit(dataMat)
    # lowDAttrMat = pca.fit_transform(dataMat)

    # if graph.node_num < ctrl.embed_dim:
    #    m = int(ctrl.embed_dim / graph.node_num)
    #    print("m = ",m)
    #    embeddings=embedding
    #    for i in range (m):
    #        embeddings=concatenate((embeddings, embedding),axis=0)
    #    print("embeddings&&&&&&&&&",embeddings.shape)
    #    pca = PCA(n_components=ctrl.embed_dim)
    #    pca.fit(embeddings)
    #    embeddings=pca.fit_transform(embeddings)
    #    embedding=embeddings[ : graph.node_num, : ]
    #    print("embedding&&&&&&&&&",embedding.shape)
    # else:
    #     pca = PCA(n_components=ctrl.embed_dim)
    #     pca.fit(embedding)
    #     embedding=pca.fit_transform(embedding)

    print("Base Embedding ok")

    multi_level_embeds = [embedding]

    # Step - 3: Embeddings Refinement.
    if ctrl.refine_model.double_base:
        coarse_embed = basic_embed(ctrl, graph.coarser)
        coarse_embed = normalized(coarse_embed, per_feature=False)
    else:
        coarse_embed = None
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=ctrl.workers)) as session:
        model = refine_model(ctrl, session)

        model.train_model(coarse_graph=graph.coarser, fine_graph=graph, coarse_embed=coarse_embed,
                         fine_embed=embedding)  # refinement model training

        while graph.finer is not None:  # apply the refinement model.
            embedding= model.refine_embedding(coarse_graph=graph, fine_graph=graph.finer, coarse_embed=embedding)
            multi_level_embeds.append(embedding)
            graph = graph.finer

    end = time.time()
    ctrl.embed_time = end - start
    print("Embedding refinement ok")
    # return embedding
    return multi_level_embeds

