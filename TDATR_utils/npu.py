import os
import copy
import sys

import torch
from functools import wraps


def wrapper_type(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        output = fn(*args, **kwargs)
        if isinstance(output, str):
            if output == 'torch.npu.FloatTensor':
                output = 'torch.cuda.FloatTensor'
            elif output == 'torch.npu.HalfTensor':
                output = 'torch.cuda.HalfTensor'
        return output

    return decorated


# deprecated
def wrapper_dist(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if args[0].dtype == torch.long and not kwargs.get('async_op', False):
            new_args = list(copy.deepcopy(args))
            new_args[0] = new_args[0].int()
            fn(*new_args, **kwargs)
            args[0].copy_(new_args[0].long())
            return
        return fn(*args, **kwargs)

    return wrapper

def set_npu(cfg=None):

    os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    torch.Tensor.type = wrapper_type(torch.Tensor.type)
    torch.distributed.all_reduce = wrapper_dist(torch.distributed.all_reduce)