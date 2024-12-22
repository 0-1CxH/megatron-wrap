# megatron-wrap

## Introducation

`megatron-wrap` is a wrap of nvidia's Megatron-LM that provides user convenience like training/inference framework of huggingface family (transformers, deepspeed, trl, ...) while fully utilize the parallelism features and speed optimizations of megatron-lm for scaling to larger model.

## Features

### Wrap

`MegatronWrap` provides easy-to-use interface of initializing megatron-lm, settting up model, train with data given at runtime (instead of the builtin dataset/loader), logging metrics and saving model, therefore you can start you project with attention paid on algorithm.

### Model and Algorithm Mangement

`MegatronWrapTrainingFlow` takes care of the data parallel and context parallel inside the data collating and loss function to avoid details of megatron-lm.

`MegatronModelProviderEntry` is a the class that organizes the model providers of megatron-lm and gives model using megatron-wrap configs.


### Config Management

The configs are organized in a tree strcute and split by the frequency of being modified across runs. The config tree supports `select`, `inherit` and `override` syntax for easier use of predefined configs and changing part of them, see docs of `confignest` for details.


### Patch

`megatron-wrap` patches logger, warning and print of megatron-lm to hide the low-level details of it, and provides user with handy utils of logging on rank_0/all_ranks.


## Quick Start

### Example

```bash
bash scripts/run_llama2_7b_minimal_mock.sh
```

### Installation

```bash
git clone https://github.com/0-1CxH/megatron-wrap.git
cd megatron-wrap
git submodule update --init --recursive # this will pull git@github.com:NVIDIA/Megatron-LM.git (core_r0.8.0) to project folder

```

If you already have a megatron-lm environment:

```
pip install -r environment/wrap_environment/requirements.txt
```

Or, see the `dependencies` section for more details.

### Fast Test

Run the builtin scripts first to check if it is installed sucessfully. 

```bash
export WANDB_API_KEY="YOUR_WANDB_API_KEY" # optional
export CONSOLE_LOG_LEVEL="INFO" # optional, default is "DEBUG"

CONFIG_FILE="PATH_TO_CONFIG_FILE"
# options are:
# "configs/llama2-7b-minimal-mock.yaml"
# "configs/llama2-7b-sft.yaml"
# better test all options
bash scripts/run_example.sh $CONFIG_FILE
```

### Training Script 

Use wrapped interface of `MegatronWrap`.

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

### Config

Here is an example (`configs/llama2-7b-minimal-mock.yaml`):

```yaml
megatron_lm:
  model:
    arch:
      __select__: llama2-7b
    parallel:
      __select__: base
      __override__:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 2
        context_parallel_size: 2
  train:
    common:
      micro_batch_size: 4
      global_batch_size: 128
      seq_length: 512
      train_iters: 64
      load: ckpt/llama-2-7b-mcore-tp2pp2
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
    __select__: minimal_mock

```

### Run

Use `torchrun` to start your script, note that `$SCRIPT` and `$CONFIG` should be the training script and config from above steps.

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


## Dependencies

Attention: read this section if you do not have a valid megatron-lm environment, the following script is executed on nvidia's image `nvcr.io/nvidia/pytorch:24.05-py3` and is just for reference, read other materials if this fails, or you can use `environment/test_environment/Dockerfile` to build a environment that this project is developed and tested in (it contains usused libs that this project does not use).

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


## Development


### inherit MegatronWrapTrainingFlowBase 

implement:

- get_fields_and_seqdims
- collate_data_micro_batch
- calculate_loss

use the following interfaces in calculate loss:

- log_each_step_metrics
- sum_on_cp_group