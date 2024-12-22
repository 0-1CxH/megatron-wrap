import torch
import torch.distributed

class SumFunction(torch.autograd.Function):
    group = None
    @staticmethod
    def forward(ctx, tensor) -> torch.Any:
       assert SumFunction.group is not None, f"need to set group first"
       torch.distributed.all_reduce(
           tensor,
           op=torch.distributed.ReduceOp.SUM,
           group=SumFunction.group
       )
       return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output