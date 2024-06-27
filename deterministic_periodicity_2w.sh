#!/bin/bash
#SBATCH --job-name=det_prd
#SBATCH --output=logs/O1_dp2w.txt
#SBATCH --error=logs/E1_dp2w.txt
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

NUM_WEEKS=20w

declare -a datasets=(er-clique-$NUM_WEEKS)
DATA_LOC=./TGB/tse/data/

cd /home/mila/a/alireza.dizaji/lab &&\
python -m TGB.tse.dataset.deterministic_periodicity -c ./TGB/tse/dataset/config/deterministic-periodicity2.yml -s ./TGB/tse/data

# # TGN scripts
# TGN_SCRIPT=TGB.examples.linkproppred.periodicity_det.tgn
# LR=1e-4
# MEM_DIM=100
# TIME_DIM=100
# EMB_DIM=100

# cd "/Users/gil-estel/Desktop/MILA/Research/TGB" 
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING TGN on $data @@@"
#     python -m $TGN_SCRIPT -d $data --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC
# done
 
# # Edge bank scripts
# # cd "/home/mila/a/alireza.dizaji/lab/TGB"
# EB_SCRIPT=TGB.examples.linkproppred.periodicity_det.edgebank
# MEM_MODE=unlimited
# for data in "${datasets[@]}"
# do
#     echo "^^^ RUNNING EDGEBANK on $data; memory mode: $MEM_MODE ^^^"
#     python -m $EB_SCRIPT --mem_mode $MEM_MODE -d $data --data-loc $DATA_LOC
# done

# # # DyRep scripts
# # DR_SCRIPT=TGB.examples.linkproppred.periodicity_det.dyrep

# # cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# # for data in "${datasets[@]}"
# # do
# #     echo "@@@ RUNNING DYREP on $data @@@"
# #     python -m $DR_SCRIPT -d $data --data-loc $DATA_LOC
# # done

# EGCNO scripts
EGCNO_SCRIPT=TGB.examples.linkproppred.periodicity_det.egcno
NODE_FEAT=ONE_HOT
NUM_UNITS=1
IN_CHANNELS=512
cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
for data in "${datasets[@]}"
do
    echo "@@@ RUNNING EvolveGCNO on $data @@@"
    echo "^^^ Number of units: $NUM_UNITS; number of channels: $IN_CHANNELS ^^^"
    python -m $EGCNO_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --num-units $NUM_UNITS --in-channels $IN_CHANNELS
done

# # GCLSTM scripts
# GCLSTM_SCRIPT=TGB.examples.linkproppred.periodicity_det.gclstm
# NODE_FEAT=ONE_HOT
# NUM_UNITS=1
# OUT_CHANNELS=512
# K=8
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING GCLSTM on $data @@@"
#     echo "^^^ Number of units: $NUM_UNITS; number of channels: $OUT_CHANNELS; Chebyshev filter size: $K ^^^"
#     python -m $GCLSTM_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --num-units $NUM_UNITS --out-channels $OUT_CHANNELS --k-gclstm $K
# done

# # HTGN scripts
# HTGN_SCRIPT=TGB.examples.linkproppred.periodicity_det.htgn
# NODE_FEAT=ONE_HOT
# OUT_CHANNELS=256
# AGG=deg
# NB_WINDOW=1
# HEADS=4
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING HTGN on $data @@@"
#     echo "^^^ number of channels: $OUT_CHANNELS; Number of windows: $NB_WINDOW; Number of heads $HEADS; Aggregation type: $AGG ^^^"
#     python -m $HTGN_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --out-channels $OUT_CHANNELS --nb-window $NB_WINDOW --heads $HEADS  --aggregation $AGG
# done

# # GCN scripts
# GCN_SCRIPT=TGB.examples.linkproppred.periodicity_det.gcn
# NODE_FEAT=ONE_HOT
# IN_CHANNELS=512
# NUM_UNITS=1
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING GCN on $data @@@"
#     echo "^^^ number of channels: $IN_CHANNELS; Number of layers: $NUM_UNITS ^^^"
#     python -m $GCN_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --in-channels $IN_CHANNELS --num-units $NUM_UNITS
# done