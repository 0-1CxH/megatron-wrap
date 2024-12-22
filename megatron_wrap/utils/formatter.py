def format_weights_info(module, prefix=''):
    s = ""
    for name, param in module.named_parameters():
        shape = param.shape
        norm = param.norm().item()
        s += f'{prefix}{name}{"[frozen]" if not param.requires_grad else ""}: shape = {shape}, norm = {norm:.4f}\n'
    return s

def format_optimizer_info(optimizer):
    config = optimizer.config
    return (f"{optimizer.__class__.__name__}(type={config.optimizer}, "
            f"lr={config.lr}, min_lr={config.min_lr}, weight_decay={config.weight_decay}"
            f"adam_beta=({config.adam_beta1},{config.adam_beta2}), adam_eps={config.adam_eps}, "
            f"sgd_momentum={config.sgd_momentum}"
            )

def format_scheduler_info(scheduler):
    s = f"{scheduler.__class__.__name__}(lr_decay_style={scheduler.lr_decay_style}, lr_warmup_steps={scheduler.lr_warmup_steps}, "
    if scheduler.lr_decay_style == "WSD":
        s += f"wsd_decay_steps={scheduler.wsd_decay_steps}"
    else:
        s += f"lr_decay_steps={scheduler.lr_decay_steps}"
    return s + ")"