import sys
from megatron_wrap.core import MegatronWrap
from megatron_wrap.utils import logger


c = MegatronWrap(sys.argv[1])
c.initialize()
c.setup_model_and_optimizer()
for _ in range(10):
    metrics = c.train([[]]*128)
    # log them first
    c.log_last_metrics()
    # log metrics to wandb

# c.save()
