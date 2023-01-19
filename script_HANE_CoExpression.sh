#!/bin/bash
#SBATCH --job-name=HANE       # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --cpus-per-task=32       # CPU cores/threads
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=400000M               # memory (per node)
#SBATCH --time=30-30:00            # time (DD-HH:MM)
#SBATCH --partition=zhanglab.p    # use zhanglab partition
#SBATCH --output=./HANE.log       # Standard output

cd ./HANE_structureOnly/

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_gcn_32/"
python main.py --data Gene_CoExpression --basic-embed grarep --refine-type gcn --embed-dim 32 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_gcn_32/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_gcn_64/"
python main.py --data Gene_CoExpression --basic-embed grarep --refine-type gcn --embed-dim 64 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_gcn_64/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_graphsage_32/"
python main.py --data Gene_CoExpression --basic-embed grarep --refine-type graphsage --embed-dim 32 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_graphsage_32/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_graphsage_64/"
python main.py --data Gene_CoExpression --basic-embed grarep --refine-type graphsage --embed-dim 64 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/grarep_graphsage_64/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_gcn_32/"
python main.py --data Gene_CoExpression --basic-embed deepwalk --refine-type gcn --embed-dim 32 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_gcn_32/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_gcn_64/"
python main.py --data Gene_CoExpression --basic-embed deepwalk --refine-type gcn --embed-dim 64 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_gcn_64/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_graphsage_32/"
python main.py --data Gene_CoExpression --basic-embed deepwalk --refine-type graphsage --embed-dim 32 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_graphsage_32/CoExpression.log"

echo -e "Running HANE on galaxy"
# mkdir "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_graphsage_64/"
python main.py --data Gene_CoExpression --basic-embed deepwalk --refine-type graphsage --embed-dim 64 > "/home/zihend1/iHerd/Data/Gene_CoExpression/Output/Embedding/deepwalk_graphsage_64/CoExpression.log"

