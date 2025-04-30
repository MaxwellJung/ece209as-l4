import torch
from torch import Tensor

weight_min_f32 = -(2**(0))
weight_max_f32 = 2**(0)
i8 = torch.iinfo(torch.int8)

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor):
        #Implement your quantization function here
        # print(f'{torch.min(input)=} {torch.max(input)=}')
        # Map input to range [0,1]
        normalized = torch.clamp((input - weight_min_f32)/(weight_max_f32-weight_min_f32), min=0, max=1)
        # Map input to range [0,255]
        q = torch.round(normalized*(i8.max-i8.min)) 
        # Map [0,255] to range [-128,127]
        q = q + i8.min
        # Map [-128,127] to range [-128/128,127/128]
        q = q/(2**7)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass the gradients are returned directly without modification.
        # This is the key step of STE
        return grad_output

# To apply the STE
def apply_ste(x):
    return StraightThroughEstimator.apply(x)

# When you want to quantize the weights, call the apply_ste function
# You need to use this function within the forward pass of your model in a custom Conv2d class.
