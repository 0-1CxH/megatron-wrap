from megatron_wrap.core import MegatronWrap
from megatron_wrap.utils import logger


c = MegatronWrap("configs/llama2-7b-sft.yaml")
c.initialize()
c.setup_model_and_optimizer()
logger.info_all_ranks(f"{c.get_parallel_states()}")
c.train([[]]*128)

# c.save()
