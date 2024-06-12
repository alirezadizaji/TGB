DATA_LOC=./TGB/tse/data/
NUM_WEEKS=100w

cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m TGB.tse.dataset.deterministic_periodicity -c ./TGB/tse/dataset/config/deterministic-periodicity.yml -s ./TGB/tse/data

# TGN scripts
TGN_SCRIPT=TGB.examples.linkproppred.periodicity_det.tgn
LR=1e-4
MEM_DIM=100
TIME_DIM=100
EMB_DIM=100
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m $TGN_SCRIPT -d er-hk-$NUM_WEEKS --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC &&\
python -m $TGN_SCRIPT -d star-tree-$NUM_WEEKS --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC &&\
python -m $TGN_SCRIPT -d cycle-path-$NUM_WEEKS --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC \

# Edge bank scripts
EB_SCRIPT=TGB.examples.linkproppred.periodicity_det.edgebank
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
MEM_MODE=fixed_time_window
TIME_WINDOW_RATIO=0.15
python -m $EB_SCRIPT --mem_mode $MEM_MODE --time_window_ratio $TIME_WINDOW_RATIO -d er-hk-$NUM_WEEKS --data-loc $DATA_LOC
python -m $EB_SCRIPT --mem_mode $MEM_MODE --time_window_ratio $TIME_WINDOW_RATIO -d star-tree-$NUM_WEEKS --data-loc $DATA_LOC
python -m $EB_SCRIPT --mem_mode $MEM_MODE --time_window_ratio $TIME_WINDOW_RATIO -d cycle-path-$NUM_WEEKS --data-loc $DATA_LOC \

# DyRep scripts
DR_SCRIPT=TGB.examples.linkproppred.periodicity_det.dyrep
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m $DR_SCRIPT -d er-hk-$NUM_WEEKS --data-loc $DATA_LOC &&\
python -m $DR_SCRIPT -d star-tree-$NUM_WEEKS --data-loc $DATA_LOC &&\
python -m $DR_SCRIPT -d cycle-path-$NUM_WEEKS --data-loc $DATA_LOC \