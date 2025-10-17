export PYTHONPATH=/data/coding/MMRec/MMRec/src:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:16:8
CUDA_LAUNCH_BLOCKING=1
python3 main.py -m MENTOR -d baby -g 0