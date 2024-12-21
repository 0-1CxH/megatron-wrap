import torch
from abc import abstractmethod
from functools import partial

from megatron_wrap.utils import logger

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

    @abstractmethod
    def collate_data_micro_batch(self, iterator): # handle cp ? 
        raise NotImplementedError
    
    @abstractmethod
    def calculate_loss(self, loss_inputs, model_forward_output): # handle cp ? 
        raise NotImplementedError
    
    def validate_model_forward_inputs(self, mf_inputs):
        return isinstance(mf_inputs, dict)

    def forward_step(self, iterator, model):
        self.current_step += 1
        model_forward_inputs, loss_inputs = self.collate_data_micro_batch(iterator)
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
        s += ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.debug_rank_0(s)


class MegatronWrapGPTModelFlow(MegatronWrapTrainingFlowBase):
    def validate_model_forward_inputs(self, mf_inputs):
        if super().validate_model_forward_inputs(mf_inputs):
            return all([k in mf_inputs for k in [
                "input_ids", "position_ids", "attention_mask"
            ]]) 


class MegatronWrapMinimalMockMSEFlow(MegatronWrapGPTModelFlow):
    def collate_data_micro_batch(self, iterator):
        import random
        mock_length = random.randint(1, self.seq_length)
        model_forward_inputs = {
            "input_ids": torch.randint(10 ,(self.micro_batch_size, mock_length)), 
            "position_ids": torch.range(0, mock_length-1, dtype=torch.int64).repeat(self.micro_batch_size).view(self.micro_batch_size, mock_length),
            "attention_mask": torch.ones(self.micro_batch_size, mock_length, dtype=torch.int64)
        }
        loss_inputs = {
            "target": torch.ones(self.micro_batch_size, mock_length)
        }
        return model_forward_inputs, loss_inputs

    def calculate_loss(self, loss_inputs, model_forward_output):
        max_values, _ = torch.max(model_forward_output, dim=-1)
        loss = torch.nn.MSELoss()(max_values, loss_inputs["target"])
        metrics = {
            "loss": loss,
            "random_length": model_forward_output.size(1)
        }
        if self.flow_config.log_each_step_metrics:
            self.log_each_step_metrics(metrics)
        return loss, metrics



class MegatronWrapFlowEntry:
    @classmethod
    def get_flow(cls, flow_config, parallel_states, micro_batch_size, seq_length):
        clz_map = {
            "minimal_mock_mse": MegatronWrapMinimalMockMSEFlow,
        }
        
        assert flow_config.flow_key in clz_map, "need to select a valid flow"
        return clz_map.get(flow_config.flow_key)(flow_config, parallel_states, micro_batch_size, seq_length)