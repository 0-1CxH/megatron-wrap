English | [中文](./README-zh.md)

# megatron-wrap
> Wrapped Megatron: As User-Friendly as HuggingFace, As Powerful as Megatron-LM 


## Introduction

`megatron-wrap` provides a wrapper for NVIDIA's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/), offering users ease of use similar to the HuggingFace series of training/inference frameworks (such as transformers, deepspeed, trl, etc.) while fully leverages Megatron-LM's parallel features and speed optimizations to scale up to larger models.

`megatron-wrap` 对NVIDIA的Megatron-LM进行了封装，对使用者提供了如同HuggingFace系列训练/推理框架一样的易用性（例如transformers、deepspeed、trl等），同时充分利用了Megatron-LM 的并行特性和速度优化，以便scale-up到更大的模型。


## Features

### Easy-to-Use Interafaces

- `megatron_wrap.core.wrap` provides easy-to-use interface of initializing megatron-lm, settting up model, train with data given at runtime (instead of the builtin dataset/loader), logging metrics and saving model, therefore you can focus your attention on developing the algorithm and avoid knowing the details about megatron-lm.
- `megatron_wrap.core.flow` abstracts the main elements of an algorithm if to be implemmented in megatron-lm, including data collating and loss calculating, the data parallelism (dp split and reduce) and context parallelism (cp split, validate and reduce) are taken care internally.

### Config Management

The configs (`megatron_wrap.core.config`) are organized in a tree strcute and split by the frequency of being modified across runs. The config tree supports `select`, `inherit` and `override` syntax (will be explained in `Quick Start` section) for easier use of predefined configs and changing part of them, see docs of `confignest` for more details.


### Patches

- `megatron_wrap.utils.dist_logger` patches the `loguru` logger with handy methods of `(error|warning_info_debug)_(rank_0|all_ranks)`
- `megatron_wrap.core.wrap` patches the python builtin `print()`(all prints goes to logger.debug_all_ranks), `logging`(removed handlers) and `warning` (hide `FutureWarning` and `UserWarning`)
- `megatron_wrap.core.wrap` patches the way of getting parallel states, instead of calling `mpu.get_(tensor_model|pipeline_model|data|context|expert_model)_parallel_(world_size|rank|group)()`, use `(t|p|d|c|e)p_(size|rank|group)` to save effort
- `megatron_wrap.utils.wandb_logger` contains a wandb wrap for the conveneince of use in logging metrics dict of both online and offline mode (replaces the megatron-lm wandb)


## Quick Start


### Step1: Install

```bash
# download this project
git clone https://github.com/0-1CxH/megatron-wrap.git
cd megatron-wrap
git submodule update --init --recursive # this will pull git@github.com:NVIDIA/Megatron-LM.git (core_r0.8.0) to project folder
```

If you have a megatron-lm environment already, just install reqs of this wrapper:

```bash
pip install -r environment/wrap_environment/requirements.txt
```

If you do not have one, see the `dependencies` section for more details.

### Step2: Test with Example

Run the builtin example script first to check if it is installed sucessfully. 

The example script is also a good starting point to make modifications on.

```bash
export WANDB_API_KEY="YOUR_WANDB_API_KEY" # optional
export CONSOLE_LOG_LEVEL="INFO" # optional, default is "DEBUG"

CONFIG_FILE="PATH_TO_CONFIG_FILE"
# options are:
# "configs/llama2-7b-minimal-mock.yaml": uses random generated tensor and mse loss to mock a training
# "configs/llama2-7b-sft.yaml": uses sft dataset (default example is 3200 sample from [trl-lib's tldr](https://huggingface.co/datasets/trl-lib/tldr) )
# better test all options
bash scripts/run_example.sh $CONFIG_FILE
```

### Step3: Write a Training Script 

Use wrapped interface of `MegatronWrap` to implement the training script, it is easier to start from the example or the following script skeleton:

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

### Step4: Write a Config File

Config includes the meagtron-lm and megatron-wrap part, both are in tree structure and meagtron-lm args will be flatten when sending to megatron. There is not many configs to change, the frequently changed parts:

- model architecture: `configs/nest/megatron_lm/model/arch`
- model parallelism: `configs/nest/megatron_lm/model/parallel`
- optimizer: `configs/nest/megatron_lm/train/optimizer`
- learning rate: `configs/nest/megatron_lm/train/learning-rate.yaml`
- common training args: `configs/nest/megatron_lm/train/common.yaml`
- computation flow (algorithm): `megatron-wrap/configs/nest/megatron_wrap/flow`
- logger: `configs/nest/megatron_wrap/logger.yaml`

The examples in the `Step2: Test with Example` uses config: 
```bash
megatron-wrap/configs/llama2-7b-minimal-mock.yaml
megatron-wrap/configs/llama2-7b-sft.yaml
```

Here is an detailed explanation of each field in the config :

```yaml
# start with configs/nest, there are two subfolders mapping to each section of args
megatron_lm: # this is the section of tree-organized meagtron-lm args
  model:
    arch: # the __confignest_manifest__ indicates you need to select one file under the folder configs/nest/megatron_lm/model/arch, error will raise if you do not make choice
      __select__: llama2-7b # use __select__ to indicate the selected file name, here the choice is configs/nest/megatron_lm/model/arch/llama2-7b.yaml 
    parallel:
      __select__: base
      __override__: # __override__ is applied, so the following items will replace the ones in the selected file
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 2
        context_parallel_size: 2
  train:
    common: # this is a file (configs/nest/megatron_lm/train/common.yaml), the following items replace the ones in file (just like the effects of __override__)
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
    __select__: gpt_sft

```

### Step5: Run with `torchrun`

Use `torchrun` to start your script, note that `$SCRIPT` and `$CONFIG` should be the training script and config from above steps.

```bash
DISTRIBUTED_ARGS="--nproc-per-node ${GPUS_PER_NODE:-8} \
                  --nnodes ${NNODES:-1} \
                  --node-rank ${NODE_RANK:-0} \
                  --master-addr ${MASTER_ADDR:-$(hostname)} \
                  --master-port ${MASTER_PORT:-22334}"

export OMP_NUM_THREADS=1
export CONSOLE_LOG_LEVEL="INFO"
export WANDB_API_KEY="xxxxxxxx"

torchrun $DISTRIBUTED_ARGS $SCRIPT $CONFIG 2>&1 | tee console.log

```


## Dependencies

Attention: read this section if you do not have a valid megatron-lm environment, the following script is executed on nvidia's image `nvcr.io/nvidia/pytorch:24.05-py3` and is just for reference, read other materials (such as the official repo of megatron-lm) if this fails, or you can use `environment/test_environment/Dockerfile` to build a environment that this project is developed and tested in (it contains usused libs that this project does not use).

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
    transformers fastchat setuptools_scm \
    wandb protobuf==3.20.3 \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.2 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN CUDACXX=/usr/local/cuda/bin/nvcc pip install \
    --force-reinstall --no-build-isolation --no-deps \
    git+https://github.com/Dao-AILab/flash-attention.git@v2.4.2 \
    git+https://github.com/huggingface/accelerate.git@v0.34.2 \
    git+https://github.com/NVIDIA/TransformerEngine.git@v1.9
```


## Development

If you want to develop a new computation flow (of an algorithm), do the following:

### Inherit from an Existing Class

The base class of all training flows is `MegatronWrapTrainingFlowBase`, in current version, the inheritage tree is:
```
+ MegatronWrapTrainingFlowBase
|____ + MegatronWrapGPTModelFlow 
      |____ - MegatronWrapMinimalMockFlow
      |____ - MegatronWrapGPTModelSFTFlow
```

If you are working with GPT model, inherit `MegatronWrapGPTModelFlow` since it contains GPT model's `validate_model_forward_inputs`, else inherit `MegatronWrapTrainingFlowBase` and implement `validate_model_forward_inputs` of the model type.

### Implement the Following Methods

- `def get_fields_and_seqdims(self) -> dict`: returns dict of `field: seqdim`, `field` is a field in the flow that contains a sequence dimension, `seqdim` is the index of the sequence dimension (for example, shape `input_ids` is [bs, seqlen] so the `seqdim` is 1)
- `def collate_data_micro_batch(self, iterator) -> dict`: the input is the data iterator of the iter() of `MegatronWrap::train` input, in this func, get a micro batch from the iterator and collate to form two dicts (`model_forward_inputs` and `loss_inputs`) of `name: tensor`, where the tensor is `torch.Tensor` on `cpu` device
- `def calculate_loss(self, loss_inputs, model_forward_output)`: this function takes the `loss_inputs` (that `collate_data_micro_batch` returns) and `model_forward_output` (result  `model.forward(model_forward_inputs)`) and returns `loss` and `metrics`, NOTE: you need to consider context parallel in this function, for the `loss_inputs` and `model_forward_output` only contains segments of this cp rank (process them separately and use `self.sum_on_cp_group` and `self.average_loss_across_dp_ranks` for aggregation)

### Register

Write a config file and put it under `megatron-wrap/configs/nest/megatron_wrap/flow`, register in `MegatronWrapFlowEntry` with the `flow_key` in the config file.


## TODO

- support starting with ray
- add model arch configs for llama, qwen, mistral, deepseek and converting scripts of hf<->mcore, might refer to [this](https://github.com/alibaba/Pai-Megatron-Patch)
- add flow of dpo(policy part), grpo(policy part) and distill(student part)
