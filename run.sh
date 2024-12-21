DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1} \
                  --node-rank ${NODE_RANK:-0} \
                  --master-addr ${MASTER_ADDR:-$(hostname)} \
                  --master-port ${MASTER_PORT:-22334}"

export OMP_NUM_THREADS=1
# export CONSOLE_LOG_LEVEL="INFO" 


torchrun $DISTRIBUTED_ARGS \
  main.py \
  2>&1 | tee console_logs
