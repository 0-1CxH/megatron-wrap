from megatron_wrap.config import MegatronWrapConfig
from megatron_wrap.utils import logger


logger.info_rank_0("main")

c = MegatronWrapConfig("configs/llama2-7b-sft.yaml")


# print(c.format_megatron_lm_args())
# print(c.format_megatron_wrap_args())
# print(c.format_all_args())