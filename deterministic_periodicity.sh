DATA_LOC=./TGB/tse/data/

cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m TGB.tse.dataset.deterministic_periodicity -c ./TGB/tse/dataset/config/deterministic-periodicity.yml -s ./TGB/tse/data

# TGN scripts
TGN_SCRIPT=TGB.examples.linkproppred.periodicity_det.tgn
LR=1e-4
MEM_DIM=100
TIME_DIM=100
EMB_DIM=100
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m $TGN_SCRIPT -d er-hk --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC &&\
python -m $TGN_SCRIPT -d star-tree --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC &&\
python -m $TGN_SCRIPT -d cycle-path --lr $LR --mem_dim $MEM_DIM --time_dim $TIME_DIM --emb_dim $EMB_DIM --data-loc $DATA_LOC \

# Edge bank scripts
EB_SCRIPT=TGB.examples.linkproppred.periodicity_det.edgebank
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m $EB_SCRIPT -d er-hk --data-loc $DATA_LOC &&\
python -m $EB_SCRIPT -d star-tree --data-loc $DATA_LOC &&\
python -m $EB_SCRIPT -d cycle-path --data-loc $DATA_LOC \

DyRep scripts
DR_SCRIPT=TGB.examples.linkproppred.periodicity_det.dyrep
cd "/Users/gil-estel/Desktop/MILA/Research/Temporal Graph" &&\
python -m $DR_SCRIPT -d er-hk --data-loc $DATA_LOC &&\
python -m $DR_SCRIPT -d star-tree --data-loc $DATA_LOC &&\
python -m $DR_SCRIPT -d cycle-path --data-loc $DATA_LOC \