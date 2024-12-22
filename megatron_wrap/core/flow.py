import math
import torch
from abc import abstractmethod
from functools import partial

from megatron_wrap.utils import logger
from megatron_wrap.utils.autograd_func import SumFunction

class MegatronWrapTrainingFlowBase:
    def __init__(self, flow_config, parallel_states, micro_batch_size, seq_length):
        self.flow_config = flow_config
        self.parallel_states = parallel_states
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.current_step = -1
    
    def __str__(self):
        return (f"{self.__class__.__name__}"
                f"(flow_key={self.flow_config.flow_key}, seq_length={self.seq_length}, micro_batch_size={self.micro_batch_size})")
    
    
    def get_fields_and_seqdims(self) -> dict:
        return {
            "input_ids": 1,
            "position_ids": 1,
            "attention_mask": 2,
            "labels": 1
        }

    def cp_split(self, data_dict):
        s = f"[WRAP] cp split on seqdim, CP{self.parallel_states.cp_rank}/{self.parallel_states.cp_size} got "
        fields_and_seqdims = self.get_fields_and_seqdims()
        for k, val in data_dict.items():
            if k in fields_and_seqdims and val is not None:
                seq_dim = fields_and_seqdims[k]
                s += f"{k}(dim{seq_dim}, {val.shape[seq_dim]}->"
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * self.parallel_states.cp_size,
                    val.shape[seq_dim] // (2 * self.parallel_states.cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([self.parallel_states.cp_rank, (2 * self.parallel_states.cp_size - self.parallel_states.cp_rank - 1)], 
                                     device="cpu", pin_memory=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                s += f"{val.shape[seq_dim]}, index={index.tolist()}) "
                data_dict[k] = val
        if self.parallel_states.tp_rank == 0  and self.parallel_states.pp_rank == 0  and \
            self.parallel_states.ep_rank == 0 and self.parallel_states.dp_rank == 0 :
            logger.debug_all_ranks(s)
        return data_dict
    
    def average_loss_across_dp_ranks(self, loss):
        from megatron.training.utils import average_losses_across_data_parallel_group
        
        if not isinstance(loss, list):
            loss = [loss]
        assert isinstance(loss, list)
        return average_losses_across_data_parallel_group(loss)[0]
        

    @abstractmethod
    def collate_data_micro_batch(self, iterator) -> dict:
        # return dict of cpu tensors
        raise NotImplementedError
    
    @abstractmethod
    def calculate_loss(self, loss_inputs, model_forward_output): # handle cp ? 
        raise NotImplementedError
    
    def validate_model_forward_inputs(self, mf_inputs):
        return isinstance(mf_inputs, dict)
    
    def validate_seqdim_divisibility(self, data_dict):
        # divisble by tp and 2 * cp
        fields_and_seqdims = self.get_fields_and_seqdims()
        for field_, seqdim_ in fields_and_seqdims.items():
            if field_ not in data_dict:
                continue
            seqlen = data_dict.get(field_).size(seqdim_)
            if not seqlen % self.parallel_states.tp_size == 0:
                logger.error_rank_0(f"dim {seqdim_} of field {field_} ({seqlen=}) should be divisable by tp size {self.parallel_states.tp_size}")
                return False
            if self.parallel_states.cp_size > 1:
                if not seqlen % (2 * self.parallel_states.cp_size) == 0:
                    logger.error_rank_0(f"dim {seqdim_} of field {field_} ({seqlen=}) should be divisable by 2 * cp size {2 * self.parallel_states.cp_size}")
                    return False
        return True
    
    def get_seqdim_divisiable_base(self):
        if self.parallel_states.cp_size == 1:
            return self.parallel_states.tp_size
        else:
            return abs(self.parallel_states.tp_size * self.parallel_states.cp_size * 2) // math.gcd(self.parallel_states.tp_size, self.parallel_states.cp_size * 2)


    def forward_step(self, iterator, model):
        self.current_step += 1
        model_forward_inputs, loss_inputs = self.collate_data_micro_batch(iterator)
        if self.validate_seqdim_divisibility(model_forward_inputs) and \
            self.validate_seqdim_divisibility(loss_inputs):
            logger.debug_rank_0(f"[WRAP] validate tp and cp, the seqlen is divisable by {self.get_seqdim_divisiable_base()}")
        else:
            raise ValueError(f"[WRAP] the seqlen should be divisable by {self.get_seqdim_divisiable_base()}")
        if self.parallel_states.cp_size > 1:
            model_forward_inputs = self.cp_split(model_forward_inputs)
            loss_inputs = self.cp_split(loss_inputs)
        assert self.validate_model_forward_inputs(model_forward_inputs) is True
        assert isinstance(loss_inputs, dict)
        self.move_tensors_to_cuda(model_forward_inputs)
        self.move_tensors_to_cuda(loss_inputs)
        model_forward_output = model(**model_forward_inputs)
        return model_forward_output, partial(self.calculate_loss, loss_inputs)
    
    @staticmethod
    def move_tensors_to_cuda(tensors: dict):
        for name, tensor in tensors.items():
            tensors[name] = tensor.to(torch.cuda.current_device())
    
    def log_each_step_metrics(self, metrics: dict):
        s = f"(micro batch step {self.current_step}) "
        s += ', '.join([f"{k}: {v:.4e}" for k, v in metrics.items()])
        logger.info_rank_0(s)
    
    def sum_on_cp_group(self, tensor):
        if self.parallel_states.cp_size > 1:
            if SumFunction.group is None:
                SumFunction.group = self.parallel_states.cp_group
            logger.debug_rank_0(f"[WRAP] auto grad reduce all (sum) of across cp ranks")
            return SumFunction.apply(tensor)
        else:
            return tensor



class MegatronWrapGPTModelFlow(MegatronWrapTrainingFlowBase):
    def validate_model_forward_inputs(self, mf_inputs):
        if super().validate_model_forward_inputs(mf_inputs):
            return all([k in mf_inputs for k in [
                "input_ids", "position_ids", "attention_mask"
            ]]) 


class MegatronWrapMinimalMockFlow(MegatronWrapGPTModelFlow):
    def get_fields_and_seqdims(self):
        d = super().get_fields_and_seqdims()
        d.update({
            "attention_mask": 1,
            "target": 1
        })
        return d
        

    def collate_data_micro_batch(self, iterator):
        import random
        mock_length = random.randint(1, self.seq_length // 4) * 4
        model_forward_inputs = {
            "input_ids": torch.randint(3 ,(self.micro_batch_size, mock_length)), 
            "position_ids": torch.range(0, mock_length-1, dtype=torch.int64).repeat(self.micro_batch_size).view(self.micro_batch_size, mock_length),
            "attention_mask": torch.ones(self.micro_batch_size, mock_length, dtype=torch.int64)
        }
        loss_inputs = {
            "target": torch.ones(self.micro_batch_size, mock_length)
        }
        return model_forward_inputs, loss_inputs

    def calculate_loss(self, loss_inputs, model_forward_output):
        max_values, _ = torch.max(model_forward_output, dim=-1)
        loss = torch.pow(max_values - loss_inputs["target"], 2).sum(-1)
        if self.parallel_states.tp_rank == 0  and self.parallel_states.pp_rank == 0  and \
            self.parallel_states.ep_rank == 0 and self.parallel_states.dp_rank == 0 :
            logger.debug_all_ranks(f"before sum on cp group: CP{self.parallel_states.cp_rank}, loss={loss.tolist()}")
        loss = self.sum_on_cp_group(loss)
        logger.debug_rank_0(f"after sum on cp group, loss={loss.tolist()}")
        loss = loss.mean()
        metrics = {
            "loss": self.average_loss_across_dp_ranks(loss),
            "random_length": model_forward_output.size(1)
        }
        if self.flow_config.log_each_step_metrics:
            self.log_each_step_metrics(metrics)
        return loss, metrics

class MegatronWrapGPTModelSFTFlow(MegatronWrapGPTModelFlow):
    def __str__(self):
        return (f"{self.__class__.__name__}"
                f"(flow_key={self.flow_config.flow_key}, seq_length={self.seq_length}, micro_batch_size={self.micro_batch_size}, collate_type={self.flow_config.collate_type})")
    

    def get_fields_and_seqdims(self):
        d = super().get_fields_and_seqdims()
        d.update(
            {
                "loss_mask": 1
            }
        )
        return d
    
    def collate_data_micro_batch(self, iterator) -> dict:
        from megatron.training.utils import get_ltor_masks_and_position_ids
        micro_batch_data = [next(iterator) for _ in range(self.micro_batch_size)]
        if self.flow_config.collate_type == "fixed_length":

            input_ids = torch.zeros(self.micro_batch_size, self.seq_length, dtype=torch.int64)
            labels = torch.zeros(self.micro_batch_size, self.seq_length, dtype=torch.int64)
            for idx, data in enumerate(micro_batch_data):
                input_ids_value = data["input_ids"]
                input_ids[idx, :len(input_ids_value)-1] = torch.tensor(input_ids_value[:-1])
                labels[idx, :len(input_ids_value)-1] = torch.tensor(input_ids_value[1:])
                
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                input_ids, 0, False, False, False
            )

            for idx, data in enumerate(micro_batch_data):
                context_token_length_value = data["context_token_length"]
                loss_mask[idx, :min(context_token_length_value, self.seq_length)] = 0
                if context_token_length_value > self.seq_length:
                    logger.warning_all_ranks(f"current data has context of {context_token_length_value}, which is larger than seqlen ({self.seq_length})")
            loss_mask = loss_mask.bool()
            loss_mask &= (labels != 0)
            
            model_forward_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
            loss_inputs = {
                "labels": labels,
                "loss_mask": loss_mask,
            }

            return model_forward_inputs, loss_inputs
                
        else:
            raise NotImplementedError("dynamic padding colalting not yet implemented")
    
    def calculate_loss(self, loss_inputs, model_forward_output):
        
        labels = loss_inputs["labels"]
        loss_mask = loss_inputs["loss_mask"]
        log_probs = torch.nn.functional.log_softmax(model_forward_output, dim=-1)
        loss = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        loss = loss.view(loss_inputs["labels"].size())
        # need to keep both loss and mask count to reduce across cp ranks
        loss *= loss_mask
        loss = loss.sum(-1)
        loss_mask_count = loss_mask.float().sum(-1)
        
        # sum both loss and mask count in cp group
        if self.parallel_states.tp_rank == 0  and self.parallel_states.pp_rank == 0  and \
            self.parallel_states.ep_rank == 0 and self.parallel_states.dp_rank == 0:
            if self.parallel_states.cp_size > 1:
                logger.debug_all_ranks(f"before sum on cp group: CP{self.parallel_states.cp_rank}, loss={loss.tolist()} loss_mask_count={loss_mask_count.tolist()}")
        
        loss = self.sum_on_cp_group(loss)
        loss_mask_count = self.sum_on_cp_group(loss_mask_count)

        if self.parallel_states.cp_size > 1:
            logger.debug_rank_0(f"after sum on cp group, loss={loss.tolist()}, loss_mask_count={loss_mask_count.tolist()}")
        else:
            if self.flow_config.log_each_step_metrics:
                logger.debug_rank_0(f"loss={loss.tolist()}, loss_mask_count={loss_mask_count.tolist()}")

        loss = loss/loss_mask_count
        loss = loss.mean()
        
        metrics = {
            "loss": self.average_loss_across_dp_ranks(loss),
            "mask_length": loss_mask_count.mean()
        }
        
        return loss, metrics


class MegatronWrapFlowEntry:
    @classmethod
    def get_flow(cls, flow_config, parallel_states, micro_batch_size, seq_length):
        clz_map = {
            "minimal_mock": MegatronWrapMinimalMockFlow,
            "gpt_sft": MegatronWrapGPTModelSFTFlow,
        }
        
        assert flow_config.flow_key in clz_map, "need to select a valid flow"
        return clz_map.get(flow_config.flow_key)(flow_config, parallel_states, micro_batch_size, seq_length)