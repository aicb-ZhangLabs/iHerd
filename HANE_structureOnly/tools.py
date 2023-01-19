import sys
import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from utils import setup_custom_logger

def getThresholdValue(matrix_Arr, quantile):
    flatten = []
    for row in matrix_Arr:
        for val in row:
            flatten.append(float(val))
    #print("Data size : ", len(flatten))
    return np.quantile(flatten, quantile)


def isEdgeExisted(edgeSet, node, adjNode):
    if ((node, adjNode) in edgeSet) or ((adjNode, node) in edgeSet):
        return True
    else:
        return False


def writeEdgelist(df, toFile, threshold_qt, logger):
    logger.info(f"Writing to {toFile}")

    #clean up data
    #df.rename({"Unnamed: 0" : "gene"}, inplace = True)

    #set up threshold value
    threshold_val = 0.1
    # if threshold_qt > 0:
    #     threshold_val = getThresholdValue(df, threshold_qt)

    f = open(toFile, "w")
    newLines = []
    edgeSet = set()
    cols = df.columns
    for i, row in tqdm(df.iterrows(), total = len(df)):
        for j, col in enumerate(cols[1:]):
            if not isEdgeExisted(edgeSet, i, j):
                edgeSet.add((i,j))
                newLines.append(str(i) + " " + str(j) + " " + str(row[col]) + "\n")
    f.writelines(newLines)
    f.close()
    return df

"""
    threshold_qt : use for potential filtering out of adjacenct nodes that has higher corr scores than threshold
"""
def parse_file(data_dir, file_name, threshold_qt = 0):
    # fromFile = os.path.join(data_dir, file_name + ".csv")
    # toFile   = os.path.join(data_dir, 'edgelist', file_name + ".edgelist")
    fromFile = data_dir + "matrices/" + file_name + ".csv"
    toFile   = data_dir + "preprocess/" + file_name + ".edgelist"
    logger   = setup_custom_logger(f"data_preprocessing_{file_name}")
    print("toFile: ", toFile)
    if os.path.exists(toFile):
        logger.info(f"{toFile} exists, skip parsing")
        return
    try:
        os.path.exists(fromFile)
    except FileNotFoundError as err:
        logger.error(err, f"{fromFile} does not exists")

    df = pd.read_csv(fromFile)

    df = writeEdgelist(df, toFile, threshold_qt, logger)

    logger.info(f"{toFile} has been written.")
