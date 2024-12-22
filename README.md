# megatron-wrap


## dependency

```bash
export MAX_JOBS=16
export CUDA_HOME='/usr/local/cuda'
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/NVIDIA/apex.git && \
    pip uninstall apex -y && \
    pushd ./apex && \
    MAX_JOBS=16 python setup.py install --cuda_ext --cpp_ext && \
    popd
pip install poetry pydantic \
    transformers datasets fastchat setuptools_scm \
    loguru wandb protobuf==3.20.3 \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.2 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN CUDACXX=/usr/local/cuda/bin/nvcc pip install \
    --force-reinstall --no-build-isolation --no-build-isolation --no-deps \
    git+https://github.com/Dao-AILab/flash-attention.git@v2.4.2 \
    git+https://github.com/huggingface/accelerate.git@v0.34.2 \
    git+https://github.com/NVIDIA/TransformerEngine.git@v1.9
```

```
repo: git@github.com:NVIDIA/Megatron-LM.git
branch: core_r0.8.0
```

```
pip install confignest==1.0.5
```


## installation

```bash
cd megatron-wrap
git submodule update --init --recursive

```

## run

```bash
bash scripts/run_llama2_7b_minimal_mock.sh 
```


```bash
DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1} \
                  --node-rank ${NODE_RANK:-0} \
                  --master-addr ${MASTER_ADDR:-$(hostname)} \
                  --master-port ${MASTER_PORT:-22334}"

export OMP_NUM_THREADS=1
export CONSOLE_LOG_LEVEL="INFO" 

export SCRIPT="main.py"
export CONFIG="configs/xxxx.yaml"

torchrun $DISTRIBUTED_ARGS $SCRIPT $CONFIG 2>&1 | tee console.log


```

