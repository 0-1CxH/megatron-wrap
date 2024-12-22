# megatron-wrap

## Introducation
This project is a wrap for nvidia's Megatron-LM to make it easier to use like training/inference framework of huggingface family (transformers, deepspeed, trl, ...) while fully utilize the parallel optimizations of megatron-lm.

## Quick Start

### write training script using wrapped interface

```python
from megatron_wrap.core import MegatronWrap

megatron_wrap = MegatronWrap(config_yaml)
megatron_wrap.initialize()
megatron_wrap.setup_model_and_optimizer()
for _ in range(train_iters):
    megatron_wrap.train(batch_data)
    megatron_wrap.log_last_metrics()
megatron_wrap.save()
```

### write a short config file



```yaml
# Example: configs/llama2-7b-minimal-mock.yaml
megatron_lm:
  model:
    arch:
      __select__: llama2-7b
    parallel:
      __select__: base
      __override__:
        tensor_model_parallel_size: 4
        pipeline_model_parallel_size: 1
  train:
    common:
      micro_batch_size: 4
      global_batch_size: 128
      seq_length: 512
      train_iters: 64
      load: ckpt/llama-2-7b-mcore-tp4pp1
      save: ckpt/llama2-7b-minimal-mock-save
      save_interval: 1
    learning-rate:
      lr: 2.0e-5
      lr_warmup_fraction: 0.05
megatron_wrap:
  init:
    megatron_lm_project_path: megatron_lm_core_080
    skip_compile_dependencies: true
  logger:
    patch_print: true
    remove_logging: true
  model_provider:
    __select__: gpt_model
    __override__:
      show_weight_details: true
  flow:
    __select__: minimal_mock_mse

```

### run with torchrun

```bash
DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1} \
                  --node-rank ${NODE_RANK:-0} \
                  --master-addr ${MASTER_ADDR:-$(hostname)} \
                  --master-port ${MASTER_PORT:-22334}"

export OMP_NUM_THREADS=1
export CONSOLE_LOG_LEVEL="INFO" 

torchrun $DISTRIBUTED_ARGS $SCRIPT $CONFIG 2>&1 | tee console.log

```

Example: scripts/run_llama2_7b_minimal_mock.sh 




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

