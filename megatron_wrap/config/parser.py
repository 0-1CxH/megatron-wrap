import os
from confignest import ConfigNest


class MegatronWrapConfig:
    def __init__(self, file_path: str):
        nest_root = os.path.join(os.path.dirname(file_path), "nest")
        self.cn = ConfigNest(nest_root, file_path)
    
    def get_megatron_lm_args(self):
        return self.cn.nest_instance.megatron_lm.export_flatten()
    
    def format_megatron_lm_args(self):
        width = 80
        s = "-" * (width//2 - 12) + " megatron-lm arguments " +  "-" * (width//2 - 12) + "\n"
        for key, value in sorted(vars(self.get_megatron_lm_args()).items()):
            pad = max(width - key.__len__() - str(value).__len__(), 0)
            s += f"{key} {'.' * pad} {value}\n"
        s += "-" * width + "\n"
        return s
    
    def get_megatron_wrap_args(self):
        return self.cn.nest_instance.megatron_wrap
    
    def format_all_args(self):
        return "-"*80 + "\n" + self.cn.format_string() + "-" * 80 + "\n"
    
    def format_megatron_wrap_args(self):
        return "megatron-wrap arguments\n" + self.get_megatron_wrap_args().format_string() + "\n"
