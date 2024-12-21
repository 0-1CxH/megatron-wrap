from abc import abstractmethod
from functools import partial

class MegatronWrapTrainingFlowBase:
    def __init__(self, flow_config, parallel_states):
        self.flow_config = flow_config
        self.parallel_states = parallel_states

    @abstractmethod
    def collate_microbatch(iterator):
        raise NotImplementedError
    
    @abstractmethod
    def loss_func(**loss_inputs_model_output):
        # model output should be the last
        pass

    def forward_func(iterator, model):
        model_forward_inputs, loss_inputs = collate_microbatch(iterator)
        output_logits = model(**model_forward_inputs)
        return output_logits, partial(loss_func, **loss_inputs)


class MegatronWrapMockTrainingFlow(MegatronWrapTrainingFlowBase):
    pass


class MegatronWrapFlowEntry:
    @classmethod
    def get_flow(cls, flow_config, parallel_states):
        clz_map = {
            ("training", "mock"): MegatronWrapMockTrainingFlow,
        }
        flow_key = (flow_config.flow_type, flow_config.flow_name)
        assert flow_key in clz_map, "need to select a valid flow"
        return clz_map.get(flow_key)(flow_config, parallel_states)