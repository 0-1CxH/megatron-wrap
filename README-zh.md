# megatron-wrap

[English](./README.md) | 中文

## 简介

`megatron-wrap` provides a wrapper for NVIDIA's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/), offering users ease of use similar to the HuggingFace series of training/inference frameworks (such as transformers, deepspeed, trl, etc.) while fully leverages Megatron-LM's parallel features and speed optimizations to scale up to larger models.

`megatron-wrap` 对NVIDIA的Megatron-LM进行了封装，对使用者提供了如同HuggingFace系列训练/推理框架一样的易用性（例如transformers、deepspeed、trl等），同时充分利用了Megatron-LM 的并行特性和速度优化，以便scale-up到更大的模型。


## 主要特性

### 易用接口

- `megatron_wrap.core.wrap` 提供了一个易于使用的接口，用于初始化megatron-lm、初始化模型、使用运行时提供的数据进行训练（而不是使用内置的数据集/加载器）、记录指标和保存模型，因此您可以专注于开发算法，而无需了解有关 megatron-lm 的细节。
- `megatron_wrap.core.flow` 抽象了在 megatron-lm 中实现算法的主要元素，包括数据整理和损失计算，数据并行性（dp 拆分和归约）以及上下文并行性（cp 拆分、验证和归约）都由内部处理

### 配置管理

配置(`megatron_wrap.core.config`) 以树状结构组织，并根据在不同运行中被修改的频率进行拆分。配置树支持 `select`, `inherit` 和 `override` 语法（将在 `快速入门` 部分进行解释），以便更轻松地使用预定义配置并更改其中的一部分，详细信息请参见 `confignest` 的文档。


### Patches

- `megatron_wrap.utils.dist_logger` 为 `loguru` 日志记录器添加了便捷的方法 `(error|warning_info_debug)_(rank_0|all_ranks)`
- `megatron_wrap.core.wrap` 修改了 Python 内置的 `print()`(所有打印输出都转到 logger.debug_all_ranks), `logging`(移除handlers) 和 `warning` (隐藏 `FutureWarning` 和 `UserWarning`)
- `megatron_wrap.core.wrap` 改进了获取并行状态的方法，无需调用 `mpu.get_(tensor_model|pipeline_model|data|context|expert_model)_parallel_(world_size|rank|group)()`, 而是使用 `(t|p|d|c|e)p_(size|rank|group)` 来节省精力
- `megatron_wrap.utils.wandb_logger` 包含一个 wandb 包装器，方便在在线和离线模式下记录指标字典（替换 megatron-lm 的 wandb）


## 快速入门

### 步骤1: 安装

```bash
# download this project
git clone https://github.com/0-1CxH/megatron-wrap.git
cd megatron-wrap
git submodule update --init --recursive # this will pull git@github.com:NVIDIA/Megatron-LM.git (core_r0.8.0) to project folder
```

如果您已经有一个 megatron-lm 环境，只需安装此包装器的依赖：

```bash
pip install -r environment/wrap_environment/requirements.txt
```

如果您没有现成的环境，请参阅“依赖项”部分以获取更多详细信息。

### 步骤2: 测试例子 

首先运行内置示例脚本以检查是否成功安装，该示例脚本也是进行修改的一个好的出发点。

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

### 步骤3：编写训练脚本

使用 `MegatronWrap` 的包装接口来实现训练脚本，可以从示例或以下脚本框架开始，这样会更容易：

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

### 步骤4：编写配置文件

配置包括 megatron-lm 和 megatron-wrap 部分，两者都是树状结构，并且在发送到 megatron 时，megatron-lm 的参数将被展平。需要更改的配置不多，经常更改的部分包括：

- model architecture: `configs/nest/megatron_lm/model/arch`
- model parallelism: `configs/nest/megatron_lm/model/parallel`
- optimizer: `configs/nest/megatron_lm/train/optimizer`
- learning rate: `configs/nest/megatron_lm/train/learning-rate.yaml`
- common training args: `configs/nest/megatron_lm/train/common.yaml`
- computation flow (algorithm): `megatron-wrap/configs/nest/megatron_wrap/flow`
- logger: `configs/nest/megatron_wrap/logger.yaml`

`步骤2: 测试例子` 中的例子使用了如下配置文件: 
```bash
megatron-wrap/configs/llama2-7b-minimal-mock.yaml
megatron-wrap/configs/llama2-7b-sft.yaml
```

以下是配置中每个字段的详细解释：

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

### 步骤5：使用 `torchrun` 运行

使用 `torchrun` 启动您的脚本，请注意，`$SCRIPT` 和 `$CONFIG` 应该是前面步骤中的训练脚本和配置。

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


## 依赖项

注意：如果您没有有效的 megatron-lm 环境，请阅读本节。以下脚本是在 NVIDIA 的镜像 `nvcr.io/nvidia/pytorch:24.05-py3` 上执行的，仅供参考。如果此方法失败，请查阅其他资料（例如 megatron-lm 的官方仓库），或者您可以使用 `environment/test_environment/Dockerfile` 来构建一个此项目开发和测试所用的环境（其中包含了一些本项目未使用的库）。

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
