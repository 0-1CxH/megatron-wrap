import builtins
import os
import sys
import torch
import gc
import time
from typing import Union
from types import SimpleNamespace

import torch.distributed
from megatron_wrap.utils.formatter import format_weights_info, format_optimizer_info, format_scheduler_info
from megatron_wrap.utils import logger, WandbWrap
from .config import MegatronWrapConfig
from .model import MegatronModelProviderEntry
from .flow import MegatronWrapFlowEntry


class MegatronWrap:
    def __init__(self, config: Union[str|MegatronWrapConfig]):
        if isinstance(config, str):
            config = MegatronWrapConfig(config)
        assert isinstance(config, MegatronWrapConfig)
        self._megatron_wrap_config = config
        self.megatron_lm_args = config.get_megatron_lm_args()
        self.megatron_wrap_args = config.get_megatron_wrap_args()
        # handle logger configs 
        self.handle_display()

        self.megatron_wrap_training_flow = None
        self.megatron_lm_prepared_for_training = False

        self.wandb_initialized = False
    
    def get_common_args(self):
        return self._megatron_wrap_config.cn.nest_instance.megatron_lm.train.common
    
    def get_flow_key(self):
        return self.megatron_wrap_args.flow.flow_key

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
    
    def handle_display(self):
        if self.megatron_wrap_args.logger.patch_print is True:
            logger.debug_rank_0(f"[PATCH] python builtin print patched, all print() will go to debug_all_ranks")
            builtins.print = logger.debug_all_ranks
        if self.megatron_wrap_args.logger.remove_logging is True:
            import logging
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logger.debug_rank_0(f"[PATCH] all logging handlers are removed, logging funcs no longer logs anything")
        if self.megatron_wrap_args.logger.ignore_warning is True:
            import warnings
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            logger.debug_rank_0(f"[PATCH] FutureWarning, UserWarning are ignored")
        
    def initialize_wandb(self):
        if os.getenv('WANDB_API_KEY') is None:
            logger.warning_rank_0(f"need to set wandb api key by export WANDB_API_KEY if use online mode")
        self.wandb_logger = WandbWrap(
            megatron_wrap_logger_args=self.megatron_wrap_args.logger, 
            run_configs=self.megatron_lm_args
        )
        logger.info_rank_0(f"wandb logger set")
        self.wandb_initialized = True

    
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
        _parallel_states = SimpleNamespace()
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
            "get_expert_model_parallel_rank": "ep_rank",

            "get_tensor_model_parallel_group": "tp_group",
            "get_pipeline_model_parallel_group": "pp_group",
            "get_data_parallel_group": "dp_group",
            "get_context_parallel_group": "cp_group",
            "get_expert_model_parallel_group": "ep_group",
        }
        for mpu_method, self_property in funcs_map.items():
            mpu_method_retval = getattr(mpu, mpu_method)()
            setattr(
                self,
                self_property,
                mpu_method_retval
            )
            setattr(
                _parallel_states,
                self_property,
                mpu_method_retval
            )
        self._parallel_states = _parallel_states
        logger.debug_rank_0(f"[PATCH] the series of get parallel state funcs are patched, use (t|p|d|c|e)p_(rank|size|group) instead of the mpu.get_(tensor_model|pipeline_model|data|context|expert_model)_parallel_(world_size|rank|group)()")
        return True
    
    def get_parallel_states(self):
        assert self.mpu_state_patched, "need to run patch_get_parallel_state first"
        return self._parallel_states

    
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
    
    def setup_model_and_optimizer(self):
        assert self.megatron_lm_is_initialized, f"need to initialize mpu first"
        from megatron.training.global_vars import get_args
        from megatron.core.enums import ModelType
        from megatron.training.training import setup_model_and_optimizer

        model_provider_args = self.megatron_wrap_args.model_provider

        model_provider = MegatronModelProviderEntry.get_provider(
            model_provider_args=model_provider_args,
            megatron_lm_args=get_args()
        )
        self.model, self.optimizer, self.opt_param_scheduler = setup_model_and_optimizer(
            model_provider,
            ModelType[model_provider_args.encoder_decoder_type].value
        )

        logger.info_rank_0(f"[STATUS] model is sucessfully built")
        if model_provider_args.show_weight_details:
            s = "\n"
            for vidx, module in enumerate(self.model):
                if len(self.model)!=1:
                    head =  '[vitual '+ vidx + ']\n'
                else:
                    head = ''
                s += f"{head}{format_weights_info(module)}"
            logger.debug_all_ranks(s)
        else:
            logger.debug_all_ranks(f"\n{self.model}")
        
        logger.info_rank_0(f"[STATUS] optimizer is sucessfully built: {format_optimizer_info(self.optimizer)}")
        logger.info_rank_0(f"[STATUS] scheduler is sucessfully built: {format_scheduler_info(self.opt_param_scheduler)}")
        return self.model, self.optimizer, self.opt_param_scheduler
    
    def _prepare_megatron_lm_for_training(self):
        assert hasattr(self, "model") and self.model is not None, f"need to init a valid model first"
        from megatron.core.utils import get_model_config
        from megatron.training.global_vars import get_args
        from megatron.core.distributed import DistributedDataParallel as DDP
        from megatron.core.distributed import finalize_model_grads

        args = get_args()
        # turn on training mode which enables dropout
        for model_module in self.model:
            model_module.train()
        
        self.iteration = 0
        self.last_metrics = None

        # setup training config
        config = get_model_config(self.model[0])
        config.grad_scale_func = self.optimizer.scale_loss
        if isinstance(self.model[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, \
                ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.model]
                if len(self.model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.delay_param_gather:
            config.param_sync_func = [lambda x: self.optimizer.finish_param_sync(model_index, x)
                                    for model_index in range(len(self.model))]
            if len(self.model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads
        self.training_configs = config

        if args.manual_gc:
            # Disable the default garbage collector and perform the collection manually.
            # This is to align the timing of garbage collection across ranks.
            assert args.manual_gc_interval >= 0, \
                'Manual garbage collection interval should be laerger than or equal to 0.'
            gc.disable()
            gc.collect()
        
        self.megatron_lm_prepared_for_training = True
    
    def _set_megatron_wrap_training_flow(self):
        from megatron.training.global_vars import get_args
        args = get_args()

        flow_config = self.megatron_wrap_args.flow
        assert flow_config.flow_type == "training", f"need to select training flow"
        self.megatron_wrap_training_flow = MegatronWrapFlowEntry.get_flow(flow_config, self.get_parallel_states(), args.micro_batch_size, args.seq_length)
        assert self.megatron_wrap_args.model_provider.model_type == self.megatron_wrap_args.flow.compatiable_model_type
        if self.megatron_wrap_training_flow is not None:
            logger.info_rank_0(f"[STATUS] successfully set megatron wrap training flow: {self.megatron_wrap_training_flow}")
        else:
            logger.error_rank_0(f"setting megatron wrap training flow failed")
    
    
    def train(self, data_batch: list):
        if self.megatron_lm_prepared_for_training is not True:
            self._prepare_megatron_lm_for_training()
        # set megatron wrap training flow
        if self.megatron_wrap_training_flow is None:
            self._set_megatron_wrap_training_flow()
        
        from megatron.training.global_vars import get_args
        from megatron.training.training import train_step
        

        args = get_args()
        assert len(data_batch) == args.global_batch_size, f"need data batch size ({len(data_batch)}) equals to global batch size ({args.global_batch_size})"
        data_batch_split_of_this_dp_rank = data_batch[self.dp_rank::self.dp_size]
        if self.tp_rank == 0 and self.pp_rank == 0 and self.ep_rank == 0 and self.cp_rank == 0:
            logger.debug_all_ranks(f"[WRAP] split batch data with dp, DP{self.dp_rank}/{self.dp_size} got {len(data_batch_split_of_this_dp_rank)} of {len(data_batch)}")

        start_time = time.time()
        metrics, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(self.megatron_wrap_training_flow.forward_step,
                       iter(data_batch_split_of_this_dp_rank),
                       self.model,
                       self.optimizer,
                       self.opt_param_scheduler,
                       self.training_configs)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        mean_time_tensor = torch.tensor(elapsed_time, device="cuda")
        torch.distributed.all_reduce(mean_time_tensor, op=torch.distributed.ReduceOp.AVG)
        metrics["elapsed_time"] =  mean_time_tensor.item()
             
        self.iteration += 1
        # reset internal micro batch step
        self.megatron_wrap_training_flow.current_step = -1

        # add more to metrics
        metrics["iteration"] = self.iteration
        metrics["consumed_samples"] = args.global_batch_size * self.iteration
        metrics["grad_norm"] = grad_norm
        metrics["num_zeros_in_grad"] = num_zeros_in_grad
        metrics["loss_scale"] = self.optimizer.get_loss_scale().item()
        for param_group in self.optimizer.param_groups:
            if param_group["is_decoupled_lr"]:
                metrics["decoupled_learning_rate"] = param_group["lr"]
            else:
                metrics["learning_rate"] = param_group["lr"]
        if args.log_params_norm:
            from megatron.training.utils import calc_params_l2_norm
            metrics["params_norm"] = calc_params_l2_norm(self.model)
        
        if args.log_throughput:
            from megatron.training.training import num_floating_point_operations
            metrics["tflops"] = num_floating_point_operations(args, args.global_batch_size) / 10**12
            metrics["throughput"] = metrics["tflops"] / (
                mean_time_tensor.item() * args.world_size
            )
        
        if self.megatron_wrap_args.logger.add_memory_to_metrics is True:
            giga_bytes = 1024.0 * 1024.0 * 1024.0
            metrics["memory_allocated"] = torch.cuda.memory_allocated() / giga_bytes
            metrics["memory_reserved"] = torch.cuda.memory_reserved() / giga_bytes
            metrics["max_memory_allocated"] = torch.cuda.max_memory_allocated() / giga_bytes
            metrics["max_memory_reserved"] = torch.cuda.max_memory_reserved() / giga_bytes
        
        if self.megatron_wrap_args.logger.add_theoretical_memoty_to_metrics is True:
            from megatron.training.theoretical_memory_usage import compute_weight_and_optimizer_memory, compute_activation_memory
            from megatron.core.num_microbatches_calculator import get_num_microbatches
            giga_bytes = 1024.0 * 1024.0 * 1024.0
            weight_and_optimizer_memory = (
                compute_weight_and_optimizer_memory(args, verbose=False) / giga_bytes
            )
            metrics["theoretical_weight_and_optimizer_memory"] = weight_and_optimizer_memory
            if not args.sequence_parallel or args.recompute_granularity != 'selective':
                pass
            else:
                activation_memory = (
                    compute_activation_memory(args, num_microbatches=get_num_microbatches(), verbose=False) / giga_bytes
                )
                metrics["theoretical_activation_memory"] = activation_memory
                total_memory = weight_and_optimizer_memory + activation_memory
                metrics["theoretical_total_memory"] = total_memory
        
        self.last_metrics = metrics
        return metrics
    
    @staticmethod
    def format_metrics(metrics):
        keys_fixed_order = ["iteration", "loss", "grad_norm", "learning_rate"]
        items = []
        for k in keys_fixed_order:
            v = metrics.get(k)
            items.append(
                f"{k} {metrics.get(k):.4e}" if isinstance(v, float) else f"{k} {metrics.get(k)}"
            )
        for k in metrics:
            if k not in keys_fixed_order:
                v = metrics.get(k)
                items.append(
                    f"{k} {metrics.get(k):.4e}" if isinstance(v, float) else f"{k} {metrics.get(k)}"
                )
                
        return " | ".join([f"{_:>8s}" for _ in items])
        
    def log_last_metrics(self):
        if self.last_metrics:
            logger.info_rank_0(self.format_metrics(self.last_metrics))
            if torch.distributed.get_rank() == 0:
                if self.wandb_initialized is False:
                    self.initialize_wandb()
                self.wandb_logger.log_metrics(self.last_metrics)
        torch.distributed.barrier()

    
    def save(self):
        assert hasattr(self, "model") and self.model is not None, f"need to init a valid model first"
        if self.megatron_lm_prepared_for_training is not True:
            self._prepare_megatron_lm_for_training()
        from megatron.training.global_vars import get_args
        from megatron.training.checkpointing import save_checkpoint
        args = get_args()

        if args.use_distributed_optimizer and args.overlap_param_gather:
            self.optimizer.disable_pre_hook()
        save_checkpoint(
            max(self.iteration-1, 0),
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            0,
        )
        if args.use_distributed_optimizer and args.overlap_param_gather:
            self.optimizer.enable_pre_hook()
        
        torch.distributed.barrier()
        logger.info_rank_0(f"[STATUS] model of iteration {self.iteration} is sucessfully saved at {os.path.join(args.save, f'iter_{self.iteration:07d}')}")

