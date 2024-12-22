DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1}"

export OMP_NUM_THREADS=1

torchrun $DISTRIBUTED_ARGS \
  main.py "configs/llama2-7b-minimal-mock.yaml" \
  2>&1 | tee console.log
