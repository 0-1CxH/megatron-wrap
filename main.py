from megatron_wrap.core import MegatronWrap
from megatron_wrap.utils import logger


c = MegatronWrap("configs/llama2-7b-sft.yaml")
c.initialize()
c.get_model_provider()()
# DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                #   --nnodes ${WORLD_SIZE} \
                #   --node_rank ${RANK} \
                #   --master_addr ${MASTER_ADDR} \
                #   --master_port ${MASTER_PORT}"
# torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0   main.py 

# print(c.format_megatron_lm_args())
# print(c.format_megatron_wrap_args())
# logger.debug_all_ranks(c.format_all_args())