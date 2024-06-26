#!/bin/bash
#SBATCH --job-name=det_prd_tune
#SBATCH --output=logs/O1_EB.txt
#SBATCH --error=logs/E1_EB.txt
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
python -m TGB.tse.dataset.deterministic_periodicity -c ./TGB/tse/dataset/config/deterministic-periodicity.yml -s ./TGB/tse/data

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

# for data in "${datasets[@]}"
# do
#     for MEM_MODE in "fixed_time_window" "unlimited"
#     do
#         for TIME_WINDOW_RATIO in 0.15 0.2 0.3
#         do
#             echo "^^^ RUNNING EDGEBANK on $data; time window ratio: $TIME_WINDOW_RATIO; memory mode: $MEM_MODE ^^^"
#             python -m $EB_SCRIPT --mem_mode $MEM_MODE --time_window_ratio $TIME_WINDOW_RATIO -d $data --data-loc $DATA_LOC
#         done
#     done
# done

# # DyRep scripts
# DR_SCRIPT=TGB.examples.linkproppred.periodicity_det.dyrep

# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING DYREP on $data @@@"
#     python -m $DR_SCRIPT -d $data --data-loc $DATA_LOC
# done

# EGCNO scripts
# EGCNO_SCRIPT=TGB.examples.linkproppred.periodicity_det.egcno
# NODE_FEAT=ONE_HOT
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING EvolveGCNO on $data @@@"
#     for NUM_UNITS in 2 4 8 16
#     do
#         for IN_CHANNELS in 64 128 256 512
#         do
#             echo "^^^ Number of units: $NUM_UNITS; number of channels: $IN_CHANNELS ^^^"
#             python -m $EGCNO_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --num-units $NUM_UNITS --in-channels $IN_CHANNELS
#         done
#     done
# done

# # GCLSTM scripts
# GCLSTM_SCRIPT=TGB.examples.linkproppred.periodicity_det.gclstm
# NODE_FEAT=ONE_HOT
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING GCLSTM on $data @@@"
#     for NUM_UNITS in 1 2 4
#     do
#         for OUT_CHANNELS in 256 512
#         do
#             for K in 1 2 4 8
#             do
#                 echo "^^^ Number of units: $NUM_UNITS; number of channels: $OUT_CHANNELS; Chebyshev filter size: $K ^^^"
#                 python -m $GCLSTM_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --num-units $NUM_UNITS --out-channels $OUT_CHANNELS --k-gclstm $K
#             done
#         done
#     done
# done

# # HTGN scripts
# HTGN_SCRIPT=TGB.examples.linkproppred.periodicity_det.htgn
# NODE_FEAT=ONE_HOT
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING HTGN on $data @@@"
#     for OUT_CHANNELS in 256 512
#     do
#         for NB_WINDOW in 1 2 4 8
#         do
#             for HEADS in 1 2 4
#             do
#                 for AGG in 'deg' 'att'
#                 do
#                     echo "^^^ number of channels: $OUT_CHANNELS; Number of windows: $NB_WINDOW; Number of heads $HEADS; Aggregation type: $AGG ^^^"
#                     python -m $HTGN_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --out-channels $OUT_CHANNELS --nb-window $NB_WINDOW --heads $HEADS  --aggregation $AGG
#                 done
#             done
#         done
#     done
# done

# # GCN scripts
# GCN_SCRIPT=TGB.examples.linkproppred.periodicity_det.gcn
# NODE_FEAT=ONE_HOT
# cd "/Users/gil-estel/Desktop/MILA/Research/TGB"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING GCN on $data @@@"
#     for IN_CHANNELS in 256 512
#     do
#         for NUM_UNITS in 1 2 4 8
#         do
#             echo "^^^ number of channels: $IN_CHANNELS; Number of layers: $NUM_UNITS ^^^"
#             python -m $GCN_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --in-channels $IN_CHANNELS --num-units $NUM_UNITS
#         done
#     done
# done