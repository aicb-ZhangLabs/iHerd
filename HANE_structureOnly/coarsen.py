# -*- coding: UTF-8 -*
from collections import defaultdict
from graph import Graph
import numpy as np
from utils import cmap2C ,graph2nx
import networkx as nx

def create_coarse_graph(stru_comms,graph):
        in_comm = stru_comms

        NewGraph=create_NewGraph(in_comm,graph)

        return NewGraph,in_comm
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def create_NewGraph(in_comm, graph):
    # in_comm = { community # : [Nodes inside this community] }
    cmap = graph.cmap
    coarse_graph_size = 0 # coarse_graph_size = the new node num for coarser graph
    for inc_idx in in_comm.keys(): # iterate through all communities
        for ele in in_comm[inc_idx]: # for each node inside a community
            cmap[ele] = coarse_graph_size # cmap : an array of size of node_num
        coarse_graph_size += 1
    newGraph=Graph(coarse_graph_size, graph.edge_num, weighted=True)
    newGraph.finer = graph
    graph.coarser = newGraph

    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt

    coarse_adj_list = newGraph.adj_list

    coarse_adj_idx = newGraph.adj_idx
    coarse_adj_wgt = newGraph.adj_wgt
    coarse_node_wgt = newGraph.node_wgt
    coarse_degree = newGraph.degree
    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    idx=0

    for idx in range(len(in_comm)):  # idx in the graph
        coarse_node_idx = idx
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list.
        group = in_comm[idx] # all nodes inside community[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]

            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]
            for j in range(istart, iend):

                k = cmap[adj_list[j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop.

                coarse_degree[coarse_node_idx] += adj_wgt[j]

        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges


    newGraph.edge_num = nedges
    newGraph.G= graph2nx(newGraph)

    newGraph.resize_adj(nedges)
    #newGraph.G=newG
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    newGraph.A = C.transpose().dot(graph.A).dot(C)  
    return newGraph
