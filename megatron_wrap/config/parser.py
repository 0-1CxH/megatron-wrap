import os
from confignest import ConfigNest

from megatron_wrap.utils import logger

class MegatronWrapConfig:
    format_width = 80
    def __init__(self, file_path: str):
        nest_root = os.path.join(os.path.dirname(file_path), "nest")
        self.cn = ConfigNest(nest_root, file_path)
        self.check_args_compatibility()
    
    def get_megatron_lm_args(self):
        return self.cn.nest_instance.megatron_lm.export_flatten()
    
    def format_megatron_lm_args(self):
        s = "-" * (self.format_width//2 - 12) + " megatron-lm arguments " +  "-" * (self.format_width//2 - 12) + "\n"
        for key, value in sorted(vars(self.get_megatron_lm_args()).items()):
            pad = max(self.format_width - key.__len__() - str(value).__len__(), 0)
            s += f"{key} {'.' * pad} {value}\n"
        s += "-" * self.format_width + "\n"
        return s
    
    def get_megatron_wrap_args(self):
        return self.cn.nest_instance.megatron_wrap
    
    def format_all_args(self):
        return "-"* self.format_width + "\n" + self.cn.format_string() + "-" * self.format_width + "\n"
    
    def format_megatron_wrap_args(self):
        return "megatron-wrap arguments\n" + self.get_megatron_wrap_args().format_string() + "\n"
    
    def check_args_compatibility(self):
        if self.get_megatron_wrap_args().output.parallel_output is True:
            if self.get_megatron_lm_args().sequence_parallel is True:
                logger.warning_rank_0(f"parallel_output is enabeld only when sequence_parallel is disabled, now setting sequence_parallel to false")
                self.cn.nest_instance.megatron_lm.model.parallel.sequence_parallel = False # set to nest, not the export ns

