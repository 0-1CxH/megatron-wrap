ARG REGISTRY=nvcr.io
ARG NGC_VERSION=24.05-py3
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
# torch version: 2.4.0a0+07cecf4
# cuda version: 12.4.1.003
# flash version: 2.4.2
FROM ${REGISTRY}/nvidia/pytorch:${NGC_VERSION} as base

##### base env setting

# timezone
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y tzdata; \
    apt clean;

# sshd
RUN mkdir /run/sshd; \
    apt install -y openssh-server sudo; \
    sed -i 's/^#\(PermitRootLogin\) .*/\1 yes/' /etc/ssh/sshd_config; \
    sed -i 's/^\(UsePAM yes\)/# \1/' /etc/ssh/sshd_config; \
    apt clean;

# entrypoint
RUN { \
    echo '#!/bin/bash -eu'; \
    echo 'ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime'; \
    echo 'echo "root:${ROOT_PASSWORD}" | chpasswd'; \
    echo 'exec "$@"'; \
    } > /usr/local/bin/entry_point.sh; \
    chmod +x /usr/local/bin/entry_point.sh;

ENV TZ Asia/Shanghai

ENV ROOT_PASSWORD root

EXPOSE 22

ENTRYPOINT ["entry_point.sh"]
CMD ["/usr/sbin/sshd", "-D", "-e"]

ENV PIP_NO_CACHE_DIR=1
ENV MAX_JOBS=16
ENV CUDA_HOME='/usr/local/cuda'

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN git clone https://github.com/NVIDIA/apex.git && \
    pip uninstall apex -y && \
    pushd ./apex && \
    MAX_JOBS=16 python setup.py install --cuda_ext --cpp_ext && \
    popd

RUN pip install poetry pydantic openai \
    transformers datasets fastchat setuptools_scm \
    ray[default] loguru wandb protobuf==3.20.3 \
    vllm==0.6.3.post1 \
    sglang==0.3.5 \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.2 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# workaround for https://github.com/microsoft/DeepSpeed/issues/1909
RUN git clone https://github.com/microsoft/DeepSpeed.git && \
    pushd ./DeepSpeed && \
    git checkout v0.15.2 && \
    rm deepspeed/ops/{csrc,op_builder} && \
    cp -R csrc op_builder deepspeed/ops/ && \
    rm deepspeed/accelerator && \
    cp -R accelerator deepspeed/ && \
    python setup.py install > /dev/null 2>&1 && \
    popd

# disable download and install torch by following packages
# with `--no-build-isolation --no-deps`
RUN CUDACXX=/usr/local/cuda/bin/nvcc pip install \
    --force-reinstall --no-build-isolation --no-build-isolation --no-deps \
    git+https://github.com/Dao-AILab/flash-attention.git@v2.4.2 \
    git+https://github.com/huggingface/accelerate.git@v0.34.2 \
    git+https://github.com/NVIDIA/TransformerEngine.git@v1.9

# sglang use flashinfer
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ && \
    apt-get install -y locales iproute2 && locale-gen en_US.UTF-8 && \
    pip uninstall pynvml -y
