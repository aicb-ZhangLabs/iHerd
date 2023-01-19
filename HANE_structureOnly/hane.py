#!/usr/bin/env python
# -*- coding: UTF-8 -*
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lv import louvainModularityOptimization
from defs import Control
from embed import multilevel_embed
from refine_model import GCN, GraphSage
from utils import read_graph, setup_custom_logger,loadDataSet,normalized
import importlib
import logging
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from classification import node_classification_F1, read_label
import time

def set_control_params(ctrl, graph, args):
    ctrl.refine_model.double_base = args.double_base
    ctrl.refine_model.learning_rate = args.learning_rate
    ctrl.refine_model.self_weight = args.self_weight

    ctrl.coarsen_level = args.coarsen_level
    ctrl.coarsen_to = max(1, graph.node_num // (2 ** ctrl.coarsen_level))  # rough estimation.
    ctrl.embed_dim = args.embed_dim
    ctrl.basic_embed = args.basic_embed
    ctrl.refine_type = args.refine_type
    ctrl.data = args.data
    ctrl.workers = args.workers
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)
    #ctrl.logger = setup_custom_logger('HANE')
    ctrl.logger = setup_custom_logger(args.data)

    if ctrl.debug_mode:
        ctrl.logger.setLevel(logging.DEBUG)
    else:
        ctrl.logger.setLevel(logging.INFO)
    #ctrl.logger.info(args)
    ctrl.logger.info(args)


def read_data(ctrl, args):
    # input_graph_path = args.data_folder + "Preprocess/" + args.subdata + ".edgelist"
    input_graph_path = args.data_folder + "PreprocessCellType/" + args.subdata + ".edgelist"
    label = []
    if args.label == True:
        input_label_path= prefix + ".label"
        label=read_label(input_label_path)
    graph = read_graph(ctrl, input_graph_path, directed=args.directed)   #utils.py
    return input_graph_path, graph, label


def select_base_embed(ctrl):
    mod_path = "base_embed_methods." + ctrl.basic_embed
    embed_mod = importlib.import_module(mod_path)
    embed_func = getattr(embed_mod, ctrl.basic_embed)
    return embed_func


def select_refine_model(ctrl):
    refine_model = None
    if ctrl.refine_type == 'gcn':
        refine_model = GCN
    elif ctrl.refine_type == 'graphsage':
        refine_model = GraphSage
    # elif ctrl.refine_type == 'MD-dumb':
    #     refine_model = GCN
    #     ctrl.refine_model.untrained_model = True
    return refine_model

def run(args):
    seed = 2022
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ctrl = Control()

    print("HANE STARTS....")
    input_graph_path, graph, y = read_data(ctrl, args)
    print("read ok")
    set_control_params(ctrl, graph, args)
    print("set ctrl ok")

    # Coarsen method
    match_method = louvainModularityOptimization

    # Base embedding
    basic_embed = select_base_embed(ctrl)

    # Refinement model
    refine_model = select_refine_model(ctrl)

    # Generate embeddings
    start = time.time()
    embeddings = multilevel_embed(ctrl, graph, match_method=match_method, basic_embed=basic_embed,
                                  refine_model=refine_model, args=args)

    end = time.time()
    print("times:", end-start)
    print("\n embeddings: \n")
    for emb in embeddings:
        print(emb.shape)

    return embeddings
