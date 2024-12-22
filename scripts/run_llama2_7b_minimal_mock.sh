DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1}"

export OMP_NUM_THREADS=1

[[ -z "${WANDB_API_KEY}" ]] && {
    echo "wandb api key is not set"
}

export WANDB_API_KEY=${WANDB_API_KEY}

torchrun $DISTRIBUTED_ARGS \
  example.py "configs/llama2-7b-minimal-mock.yaml" \
  2>&1 | tee console.log
