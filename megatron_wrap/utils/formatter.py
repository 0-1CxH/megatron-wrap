def format_weights_info(module, prefix=''):
    s = ""
    for name, param in module.named_parameters():
        shape = param.shape
        norm = param.norm().item()
        s += f'{prefix}{name}{"[frozen]" if not param.requires_grad else ""}: shape = {shape}, norm = {norm:.4f}\n'
    return s
