import builtins
import os
import sys
import torch
import gc
from typing import Union
from megatron_wrap.utils import logger
from .config import MegatronWrapConfig
from .model import MegatronModelProviderEntry


class MegatronWrap:
    def __init__(self, config: Union[str|MegatronWrapConfig]):
        if isinstance(config, str):
            config = MegatronWrapConfig(config)
        assert isinstance(config, MegatronWrapConfig)
        self._megatron_wrap_config = config
        self.megatron_lm_args = config.get_megatron_lm_args()
        self.megatron_wrap_args = config.get_megatron_wrap_args()
        # handle logger configs 
        self.patch_print()
        self.remove_logging()

    def initialize(self):
        logger.info_rank_0("[STATUS] initialization started")
        # dynamic import megatron
        self.megatron_lm_is_importable = self.dynamic_import_magatron_lm()
        self.megatron_lm_is_initialized = self.megatron_lm_initialize()
        # easier get *p size/rank
        self.mpu_state_patched = self.patch_get_parallel_state()
        logger.info_all_ranks("[STATUS] initialization finished, parallel state of this rank: " + self.format_parallel_states())


    def dynamic_import_magatron_lm(self):
        megatron_lm_project_absolute_path = os.path.abspath(self.megatron_wrap_args.init.megatron_lm_project_path)
        sys.path.append(megatron_lm_project_absolute_path)
        logger.debug_rank_0(f"[WRAP] added '{megatron_lm_project_absolute_path}' to sys.path")
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1" # need gradient_accumulation
        try:
            import megatron
            logger.debug_all_ranks(f"[STATUS] megatron-lm imported successfully")
            return True
        except Exception as e:
            logger.error_all_ranks(f"failed to import megatron-lm from {megatron_lm_project_absolute_path}, due to: {e}")
            return False
    
    def patch_print(self):
        if self.megatron_wrap_args.logger.patch_print is True:
            logger.debug_rank_0(f"[PATCH] python builtin print patched, all print() will go to debug_all_ranks")
            builtins.print = logger.debug_all_ranks
    
    def remove_logging(self):
        if self.megatron_wrap_args.logger.remove_logging is True:
            import logging
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logger.debug_rank_0(f"[PATCH] all logging handlers are removed, logging funcs no longer logs anything")
    
    def megatron_lm_initialize(self):
        assert self.megatron_lm_is_importable, f"need to import megatron first"
        # dynamic import
        import megatron
        from megatron.training.checkpointing import load_args_from_checkpoint
        # patch print args
        megatron.training.arguments._print_args = lambda *args: logger.info_rank_0(self._megatron_wrap_config.format_megatron_lm_args())
        from megatron.training.arguments import validate_args 
        from megatron.training.global_vars import set_global_variables, get_args
        from megatron.training.initialize import (
            _init_autoresume,
            _initialize_distributed,
            _initialize_tp_communicators,
            _set_random_seed,
            _compile_dependencies,
            set_jit_fusion_options,
        )


        if not self.megatron_wrap_args.init.allow_no_cuda:
            assert torch.cuda.is_available(), "megatron-lm requires CUDA"
        
        args = self.megatron_lm_args
        if args.use_checkpoint_args:
            assert args.load is not None, "if use checkpoints args, need valid --load argument"
            load_args_from_checkpoint(args)
        
        validate_args(args, {})
        set_global_variables(args, build_tokenizer=False)
        
        if not args == self.megatron_lm_args == get_args():
            logger.warning_rank_0(f"there are args changed during validation, please double check")

        logger.debug_rank_0(f"[STATUS] initializing torch.distributed")
        _initialize_distributed()
        
        logger.debug_rank_0(f"setting random seeds: {args.seed}") # random seeds for reproducibility
        _set_random_seed(args.seed, args.data_parallel_random_init)
        
        logger.debug_rank_0(f"[STATUS] initializing auto resume")
        _init_autoresume()

        if not self.megatron_wrap_args.init.skip_compile_dependencies:
            logger.debug_rank_0(f"[STATUS] compiling dependencies and loading fused kernels")
            _compile_dependencies()
        
        if args.tp_comm_overlap:
            _initialize_tp_communicators()
        
        if not self.megatron_wrap_args.init.skip_set_jit_fusion:
            # this is mainly for training, slow down if set jit fusion for inference model
            logger.debug_rank_0(f"[STATUS] setting jit fusion options")
            set_jit_fusion_options()
        
        if not args == self.megatron_lm_args == get_args():
            logger.warning_rank_0(f"there are args changed during init, please double check")

        return True
    
    def patch_get_parallel_state(self):
        assert self.megatron_lm_is_initialized, f"need to initialize mpu first"
        from megatron.core import mpu
        # "tp-cp-ep-dp-pp"
        funcs_map = {
            "get_tensor_model_parallel_world_size": "tp_size",
            "get_pipeline_model_parallel_world_size": "pp_size",
            "get_data_parallel_world_size": "dp_size",
            "get_context_parallel_world_size": "cp_size",
            "get_expert_model_parallel_world_size": "ep_size",

            "get_tensor_model_parallel_rank": "tp_rank",
            "get_pipeline_model_parallel_rank": "pp_rank",
            "get_data_parallel_rank": "dp_rank",
            "get_context_parallel_rank": "cp_rank",
            "get_expert_model_parallel_rank": "ep_rank"
        }
        for mpu_method, self_property in funcs_map.items():
            setattr(
                self,
                self_property,
                getattr(mpu, mpu_method)()
            )
        logger.debug_rank_0(f"[PATCH] the series of get parallel state funcs are patched, use (t|p|d|c|e)p_(rank|size) instead of the original to save effort")
        return True
    
    def format_parallel_states(self):
        assert self.mpu_state_patched, "need to run patch_get_parallel_state first"
        ts = [
            f"TP{self.tp_rank}/{self.tp_size}",
            f"PP{self.pp_rank}/{self.pp_size}",
            f"DP{self.dp_rank}/{self.dp_size}",
            f"CP{self.cp_rank}/{self.cp_size}",
            f"EP{self.ep_rank}/{self.ep_size}",
        ]
        return "(" + " ".join(ts) + ")"

    
    @classmethod
    def empty_cuda_cache(cls):
        logger.debug_all_ranks(
            f"starting empty cache on cuda device:{torch.cuda.current_device()}:"
            f"{torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.2f} GiB."
        )

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        logger.debug_all_ranks(
            f"ending empty cache on cuda device:{torch.cuda.current_device()}:"
            f"{torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.2f} GiB."
        )
    
    def get_model_provider(self):
        assert self.megatron_lm_is_initialized, f"need to initialize mpu first"
        from megatron.training.global_vars import get_args
        return MegatronModelProviderEntry.get_provider(
            model_provider_args=self.megatron_wrap_args.model_provider,
            megatron_lm_args=get_args()
        )
    
