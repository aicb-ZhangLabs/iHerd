import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import hane
import tools
import pickle
import glob
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='Gene_CoExpression', \
                        choices=['GM12878_K562_TF2TF', 'Gene_CoExpression', 'Gene_CoExpression_CellType', 'PEC2_TF2TF'],
                        help='Input graph file')
    parser.add_argument('--format', required=False, default='edgelist', choices=['metis', 'edgelist'],
                        help='Format of the input graph file (metis/edgelist)')
    parser.add_argument('--no-eval', action='store_true',
                        help='Evaluate the embeddings.')
    parser.add_argument('--embed-dim', default=32, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--basic-embed', required=False, default='grarep',
                        choices=['deepwalk', 'grarep', 'netmf','stne'],
                        help='The basic embedding method. If you added a new embedding method, please add its name to choices')
    parser.add_argument('--refine-type', required=False, default='graphsage',
                        choices=['gcn', 'dumb', 'graphsage'],
                        help='The method for refining embeddings.')
    parser.add_argument('--coarsen-level', default=2, type=int,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers.')
    parser.add_argument('--double-base', action='store_true',
                        help='Use double base for training')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Learning rate of the refinement model')
    parser.add_argument('--self-weight', default=0.05, type=float,
                        help='Self-loop weight for GCN model.')  # usually in the range [0, 1]
    parser.add_argument('--directed', default=False,                   
                        action='store_true', help='treat graph as directed')
    parser.add_argument('--label', required = False, default = False, # If ./dataset has '.label'
                        action = 'store_true', help='Include label for testing')
    parser.add_argument('--threshold_qt', default = 0.9, type = int,
                        help='Setting threshold value at quantile')
    parser.add_argument('--data-folder', required=False, default = "../../Data/GM12878_K562_TF2TF/",
                        help = 'Fold contained dataset and embedding directories, close with "/" ')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.data == "GM12878_K562_TF2TF":
        args.data_folder = "../../Data/GM12878_K562_TF2TF/"
        args.directed = True

        samples = ["GM12878", "k562"]
        # samples = ["GM12878"]
        # samples = ["k562"]

        embedding_path = args.data_folder + "Output/Embedding/"+\
                            args.basic_embed+"_"+ \
                            args.refine_type+"_"+str(args.embed_dim)+"/"
        
        if not os.path.exists(embedding_path):
           # Create a new directory because it does not exist
           os.makedirs(embedding_path)

        for sample in samples:
            print(sample)
            args.subdata = sample 
            embedding_file = embedding_path + args.subdata + ".pkl"
            embed = hane.run(args)
            with open(embedding_file, "wb") as f:
                pickle.dump(embed, f, -1)

    elif args.data == "Gene_CoExpression":
        args.data_folder = "../../Data/Gene_CoExpression/"
        args.directed = False

        samples = []
        for file in os.listdir(args.data_folder+"Raw/CoExpression/"):
            samples.append(file.replace(".csv",""))

        embedding_path = args.data_folder + "Output/Embedding/"+\
                            args.basic_embed+"_"+ \
                            args.refine_type+"_"+str(args.embed_dim)+"/"
        
        if not os.path.exists(embedding_path):
           # Create a new directory because it does not exist
           os.makedirs(embedding_path)

        for sample in samples:
            print(sample)
            args.subdata = sample
            embedding_file = embedding_path + args.subdata + ".pkl"
            embed = hane.run(args)
            with open(embedding_file, "wb") as f:
                pickle.dump(embed, f, -1)

    elif args.data == "Gene_CoExpression_CellType":
        args.data_folder = "../../Data/Gene_CoExpression/"
        args.directed = False

        samples = ["CON_EXC", "CON_MIC"]

        embedding_path = args.data_folder + "OutputCellType/Embedding/"+\
                            args.basic_embed+"_"+ \
                            args.refine_type+"_"+str(args.embed_dim)+"/"
        
        if not os.path.exists(embedding_path):
           # Create a new directory because it does not exist
           os.makedirs(embedding_path)

        for sample in samples:
            print(sample)
            args.subdata = sample
            embedding_file = embedding_path + args.subdata + ".pkl"
            embed = hane.run(args)
            with open(embedding_file, "wb") as f:
                pickle.dump(embed, f, -1)

    elif args.data == "PEC2_TF2TF":
        args.data_folder = "../../Data/PEC2_TF2TF/"
        args.directed = True
        cell_types = []
        for file in os.listdir(args.data_folder+"Raw/"):
            cell_types.append(file.replace(".TF-TF.txt",""))

        embedding_path = args.data_folder + "Output/Embedding/"+\
                            args.basic_embed+"_"+ \
                            args.refine_type+"_"+str(args.embed_dim)+"/"
        
        if not os.path.exists(embedding_path):
           # Create a new directory because it does not exist
           os.makedirs(embedding_path)

        for cell_type in cell_types:
            print(cell_type)
            args.subdata = cell_type
            embedding_file = embedding_path + args.subdata + ".pkl"
            embed = hane.run(args)
            with open(embedding_file, "wb") as f:
                pickle.dump(embed, f, -1)
    