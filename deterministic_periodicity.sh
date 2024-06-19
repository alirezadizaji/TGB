#!/bin/bash
NUM_WEEKS=2w

declare -a datasets=(er-clique-$NUM_WEEKS)
DATA_LOC=./TGB/tse/data/

cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
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
EGCNO_SCRIPT=TGB.examples.linkproppred.periodicity_det.egcno
NODE_FEAT=ONE_HOT
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph"
for data in "${datasets[@]}"
do
    echo "@@@ RUNNING EvolveGCNO on $data @@@"
    for NUM_UNITS in 2
    do
        for IN_CHANNELS in 512
        do
            echo "^^^ Number of units: $NUM_UNITS; number of channels: $IN_CHANNELS ^^^"
            python -m $EGCNO_SCRIPT -d $data --data-loc $DATA_LOC --node-feat $NODE_FEAT --num-units $NUM_UNITS --in-channels $IN_CHANNELS
        done
    done
done