CONFIG_FILE=$1

[[ -z "${CONFIG_FILE}" ]] && {
    echo "need to specify a config file by exporting CONFIG_FILE"
    exit
}

echo "using config file ${CONFIG_FILE}"

[[ -z "${WANDB_API_KEY}" ]] && {
    echo "warning: wandb api key is not set"
}

DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1} \
                  --node-rank ${NODE_RANK:-0} \
                  --master-addr ${MASTER_ADDR:-$(hostname)} \
                  --master-port ${MASTER_PORT:-22334}"

export OMP_NUM_THREADS=1

# export CONSOLE_LOG_LEVEL="INFO"

export WANDB_API_KEY=${WANDB_API_KEY}

torchrun $DISTRIBUTED_ARGS \
  example.py $CONFIG_FILE \
  2>&1 | tee console.log
