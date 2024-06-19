#!/bin/bash
#SBATCH --job-name=det_periodicity
#SBATCH --output=logs/O_dp1.txt
#SBATCH --error=logs/E_dp1.txt
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

# cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" 
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING TGN on $data @@@"
#     python -m $TGN_SCRIPT -d $data --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC
# done
 
# # Edge bank scripts
# EB_SCRIPT=TGB.examples.linkproppred.periodicity_det.edgebank
# MEM_MODE=fixed_time_window
# TIME_WINDOW_RATIO=0.15

# cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING EDGEBANK on $data @@@"
#     python -m $EB_SCRIPT --mem_mode $MEM_MODE --time_window_ratio $TIME_WINDOW_RATIO -d $data --data-loc $DATA_LOC
# done

# # DyRep scripts
# DR_SCRIPT=TGB.examples.linkproppred.periodicity_det.dyrep

# cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph"
# for data in "${datasets[@]}"
# do
#     echo "@@@ RUNNING DYREP on $data @@@"
#     python -m $DR_SCRIPT -d $data --data-loc $DATA_LOC
# done

# EGCNO scripts
# EGCNO_SCRIPT=TGB.examples.linkproppred.periodicity_det.egcno
# NODE_FEAT=ONE_HOT
# cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph"
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

# GCLSTM scripts
GCLSTM_SCRIPT=TGB.examples.linkproppred.periodicity_det.gclstm
NODE_FEAT=ONE_HOT
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph"
for data in "${datasets[@]}"
do
    echo "@@@ RUNNING GCLSTM on $data @@@"
    for NUM_UNITS in 1 2 4
    do
        for OUT_CHANNELS in 256 512
        do
            for K in 1 2 4 8
            do
                echo "^^^ Number of units: $NUM_UNITS; number of channels: $OUT_CHANNELS; Chebyshev filter size: $K ^^^"
                python -m $GCLSTM_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --num-units $NUM_UNITS --out-channels $OUT_CHANNELS --k-gclstm $K
            done
        done
    done
done