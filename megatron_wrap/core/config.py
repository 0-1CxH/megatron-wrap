import os
from confignest import ConfigNest

from megatron_wrap.utils import logger

class MegatronWrapConfig:
    format_width = 80
    def __init__(self, file_path: str):
        nest_root = os.path.join(os.path.dirname(file_path), "nest")
        self.cn = ConfigNest(nest_root, file_path)
        self.check_args_compatibility()
        self.add_runtime_args()
    
    def get_megatron_lm_args(self):
        return self.cn.nest_instance.megatron_lm.export_flatten()
    
    def format_megatron_lm_args(self):
        s = "\n" +  "-" * (self.format_width//2 - 12) + " megatron-lm arguments " +  "-" * (self.format_width//2 - 12) + "\n"
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
        if self.cn.nest_instance.megatron_wrap.model_provider.parallel_output is False:
            if self.cn.nest_instance.megatron_lm.model.parallel.sequence_parallel is True:
                logger.warning_rank_0(f"parallel_output is disabeld only when sequence_parallel is disabled, now setting sequence_parallel to false")
                self.cn.nest_instance.megatron_lm.model.parallel.sequence_parallel = False # set to nest, not the export ns
        if self.cn.nest_instance.megatron_lm.misc.other.yaml_cfg is not None:
            self.cn.nest_instance.megatron_lm.misc.other.yaml_cfg = None
            logger.warning_rank_0(f"yaml_cfg is set to null and ignored in initialization")
        if self.cn.nest_instance.megatron_lm.train.dist_comm.lazy_mpu_init is not None:
            self.cn.nest_instance.megatron_lm.train.dist_comm.lazy_mpu_init = None
            logger.warning_rank_0(f"lazy_mpu_init is set to null and ignored in initialization")
        
        if self.cn.nest_instance.megatron_wrap.logger.add_params_norm_to_metrics is False:
            self.cn.nest_instance.megatron_lm.misc.log.log_params_norm = False
            logger.warning_rank_0(f"add_params_norm_to_metrics is disabled, now setting log_params_norm to false")
        if self.cn.nest_instance.megatron_wrap.logger.add_throughput_to_metrics is False:
            self.cn.nest_instance.megatron_lm.misc.log.log_throughput = False
            logger.warning_rank_0(f"add_throughput_to_metrics is disabled, now setting log_throughput to false")
        
        # pad vocab size (if not already set from a checkpoint)
        # divisible by model parallel size and still having GPU friendly size.
        def pad_vocab_size():
            current_size = self.cn.nest_instance.megatron_lm.model.arch.vocab_size
            divide_by = self.cn.nest_instance.megatron_lm.model.arch.make_vocab_size_divisible_by * self.cn.nest_instance.megatron_lm.model.parallel.tensor_model_parallel_size
            while current_size % divide_by != 0:
                current_size += 1
            if current_size != self.cn.nest_instance.megatron_lm.model.arch.vocab_size:
                logger.warning_rank_0(f"the vocab size {self.cn.nest_instance.megatron_lm.model.arch.vocab_size} should be divisable by {divide_by} and hence padded to {current_size}")
            return current_size
        setattr(self.cn.nest_instance.megatron_lm.model.arch, "padded_vocab_size", pad_vocab_size())
        logger.debug_rank_0(f"padded_vocab_size is set to {self.cn.nest_instance.megatron_lm.model.arch.padded_vocab_size}")
        
    
    def add_runtime_args(self):
        rank = int(os.environ.get("RANK"))
        local_rank = int(os.environ.get("LOCAL_RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        self.cn.nest_instance.megatron_lm.train.dist_comm.rank = rank
        self.cn.nest_instance.megatron_lm.train.dist_comm.local_rank = local_rank
        self.cn.nest_instance.megatron_lm.train.dist_comm.world_size = world_size
