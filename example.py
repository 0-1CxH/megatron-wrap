import sys
from megatron_wrap.core import MegatronWrap
from megatron_wrap.utils import logger

from megatron_wrap.utils.dataset import EmptyDataset

config_file = sys.argv[1]
wrap = MegatronWrap(config_file)

if wrap.get_flow_key() == "minimal_mock":
    dataset = EmptyDataset()
else:
    raise ValueError()
train_iters = wrap.get_common_args().train_iters
save_interval = wrap.get_common_args().save_interval
global_batch_size = wrap.get_common_args().global_batch_size

logger.info_rank_0(f"using config {config_file}, train for {train_iters} iters, save after each {save_interval} iters")

wrap.initialize()
wrap.setup_model_and_optimizer()

for current_iteration in range(train_iters+1):
    metrics = wrap.train(dataset.get_batch(global_batch_size))
    wrap.log_last_metrics()
    # log metrics to wandb
    if save_interval is not None:
        if (current_iteration % save_interval == 0 and current_iteration > 0) or (current_iteration==train_iters):
            wrap.save()

