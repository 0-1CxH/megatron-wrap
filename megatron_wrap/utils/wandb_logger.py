import os
import wandb

class WandbWrap:
    def __init__(self, megatron_wrap_logger_args, run_configs):
        self.megatron_wrap_logger_args = megatron_wrap_logger_args
        self.enabled = self.megatron_wrap_logger_args.enable_wandb
        if self.enabled:
            online_mode = False
            api_key = os.getenv('WANDB_API_KEY')
            if api_key is not None:
                wandb.login(key=api_key)
                online_mode = True
            kwargs = {
                "dir": megatron_wrap_logger_args.wandb_dir,
                "name": megatron_wrap_logger_args.wandb_name,
                "mode": "online" if online_mode else "offline",
                "config": run_configs,
                "project": megatron_wrap_logger_args.wandb_project,
                "entity": megatron_wrap_logger_args.wandb_entity
            }
            wandb.init(
                **kwargs
            )
    
    def log_metrics(self, metrics, iteration_key="iteration"):
        if self.enabled:
            _m = metrics.copy()
            iteration = _m.pop(iteration_key)
            wandb.log(_m, iteration)
        
