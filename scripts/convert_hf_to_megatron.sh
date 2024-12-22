#!/bin/bash
set -x

# check the path
[[ -z "${MEGATRON}" ]] && {
    echo "MEGATRON path is not set"
    exit 1
}

SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
export PYTHONPATH=${SITE_PACKAGES}:${MEGATRON}:${PYTHONPATH}

export CUDA_DEVICE_MAX_CONNECTIONS=1

export TP=${TP:-2}
export PP=${PP:-2}
export LOAD_DIR=${LOAD_DIR:-"ckpt/llama-2-7b-hf"}
export SAVE_DIR=${SAVE_DIR:-"ckpt/llama-2-7b-mcore-tp${TP}pp${PP}"}
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-"sample/llama2_32k_tokenizer_files/tokenizer.model"}
export MODEL_SIZE=${MODEL_SIZE:-llama2-7B}

set +x

pushd ${MEGATRON}
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --checkpoint-type hf \
    --model-size ${MODEL_SIZE} \
    --saver mcore \
    --target-tensor-parallel-size ${TP} \
    --target-pipeline-parallel-size ${PP} \
    --load-dir ${LOAD_DIR} \
    --save-dir ${SAVE_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}
popd
