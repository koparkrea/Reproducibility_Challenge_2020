import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
device = torch.device('cuda:2')
torch.cuda.set_device(2)

def get_model_info(model, shape):
    model = add_flops_counting_method(model)
    model.eval()

    cache_inputs = torch.rand(*shape)
    if next(model.parameters()).is_cuda:
        cache_inputs = cache_inputs.to(device)
    with torch.no_grad():
        __ = model(cache_inputs)
    
    FLOPS = compute_average_flops_cost(model) / 1e6
    Param = count_parameters_in_MB(model)

    if hasattr(model, 'auxiliary_param'):
        aux_param = count_parameters_in_MB(mode.auxiliary_param())
        print('the auxiliary params of this model is : {:}'.format(aux_param))
        Param = Param - aux_param
    
    torch.cuda.empty_cache()
    model.apply(remove_hook_function)
    return FLOPS, Param




def add_flops_counting_method(model):
    model.__batch_counter__ = 0
    add_batch_counter_hook_function(model)
    model.apply(add_flops_counter_variable_or_reset)
    model.apply(add_flops_counter_hook_function)
    return model

def compute_average_flops_cost(model):
    batches_count = model.__batch_counter__
    flops_sum = 0

    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or hasattr(module, 'calculate_flop_self'):
            flops_sum += module.__flops__
    return flops_sum / batches_count


def count_parameters_in_MB(model):
    if isinstance(model, nn.Module):
        return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    else:
        return np.sum(np.prod(v.size()) for v in model) / 1e6


def pool_flops_counter_hook(module, inputs, output):
    batch_size = inputs[0].size(0)
    kernel_size = module.kernel_size
    out_C, output_height, output_width = output.shape[1:]

    overall_flops = batch_size * out_C * output_height * output_width * kernel_size * kernel_size
    module.__flops__ += overall_flops



def conv2d_flops_counter_hook(module, inputs, output):
    batch_size = inputs[0].size(0)
    output_height, output_width = output.shape[2:]
    kernel_height, kernel_width = module.kernel_size
    in_ch = module.in_channels
    out_ch = module.out_channels
    groups = module.groups
    conv_per_flops = kernel_height * kernel_width * in_ch * out_ch / groups
    active_elements = batch_size * output_height * output_width
    overall_flops = conv_per_flops * active_elements

    if module.bias is not None:
        overall_flops += out_ch * active_elements
    module.__flops__ += overall_flops


def fc_flops_counter_hook(module, inputs, output):
    batch_size = inputs[0].size(0)
    xin, xout = module.in_features, module.out_features
    assert xin == inputs[0].size(1) and xout == output.size(1), 'IO = ({:},{:})'.format(xin,xout)
    overall_flops = batch_size * xin * xout
    if module.bias is not None:
        overall_flops += batch_size * xout
    module.__flops__ += overall_flops

def batch_counter_hook(module, inputs, output):
    inputs = inputs[0]
    batch_size = inputs.shape[0]
    module.__batch_counter__+=batch_size



def add_batch_counter_hook_function(model):
    if not hasattr(model, '__batch_counter_handel__'):
        handle = model.register_forward_hook(batch_counter_hook)
        model.__batch_counter_handel__ = handle


def add_flops_counter_variable_or_reset(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d) or hasattr(module, 'calculate_flop_self'):
        module.__flops__ = 0

def add_flops_counter_hook_function(module):
    if isinstance(module, nn.Conv2d):
        if not hasattr(module, '__flops_handle__'):
            handle = module.register_forward_hook(conv2d_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, nn.Linear):
        if not hasattr(module, '__flops_handle__'):
            handle = module.register_forward_hook(fc_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d):
        if not hasattr(module, '__flops_handle__'):
            handle = module.register_forward_hook(pool_flops_counter_hook)
            module.__flops_handle__ = handle
    elif hasattr(module, 'calculater_flop_self'):
        if not hasattr(module, '__flops_handle__'):
            handle = module.register_forward_hook(self_calculate_flops_counter_hook)
            module.__flops_handle__ = handle


def remove_hook_function(module):
    hookers = ['__batch_counter_handle__', '__flops_handle__']
    for hooker in hookers:
        if hasattr(module, hooker):
            handle = getattr(module, hooker)
            handle.remove()
    keys = ['__flops__', '__batch_counter__', '__flops__'] + hookers
    for key in keys:
        if hasattr(module, key):
            delattr(module, key)
    module.__dict__['_forward_hooks'] = OrderedDict()
