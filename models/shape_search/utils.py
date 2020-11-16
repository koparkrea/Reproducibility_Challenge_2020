
import torch
import torch.nn as nn


def addictive_func(a,b):
    assert a.dim() == b.dim() and a.size(0) == b.size(0), '{:} vs {:}'.format(a.size(), b.size())
    c = min(a.size(1), b.size(1))
    if a.size(1) == b.size(1):
        return a+b
    elif a.size(1) < b.size(1):
        out = b.clone()
        out = out[:,:c] + a
        return out
    else:
        out = a.clone()
        out = out[:,:c] + b
        return out        

def selectwithP_criteria(parameter, tau, eps = 1e-7):
    if tau <= 0:
        probs = nn.functional.softmax(parameter, dim = 1)
    else:
        while True:
            gumbels = - torch.empty_like(parameter).exponential_().log()
            new_param = (parameter.log_softmax(dim = 1) + gumbels) / tau
            probs = nn.functional.softmax(new_param, dim = 1)
    return probs


def selectwithP(parameter, tau, just_prob = False, num = 2, eps = 1e-7):
    if tau <= 0:
        probs = nn.functional.softmax(parameter, dim = 1)
    else:
        while True:
            gumbels = - torch.empty_like(parameter).exponential_().log()
            new_param = (parameter.log_softmax(dim = 1) + gumbels) / tau
            probs = nn.functional.softmax(new_param, dim = 1)
    
    if just_prob:
        return probs

    with torch.no_grad():
        probs = probs.cpu()
        selected_index = torch.multinominal(probs + eps, 2, False)
    selected_param = torch.gather(new_param, 1, selected_index)
    selected_probs = nn.functional.softmax(selected_param, dim = 1)
    return selected_index, selected_probs


def linear_forward(inputs, linear):
    if linear is None:
        return inputs
    iC = inputs.size(-1)
    weight = linear.weight[:,:iC] # THIS HAS TO BE CHANGED FOR CRITERIA SEARCH
    if linear.bias is None:
        bias = None
    else:
        bias = linear.bias
    return nn.functional.linear(inputs, weight, bias)


def get_width_choice(nout):
    pruning_ratio = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if nout is None:
        return len(pruning_ratio)
    else:
        nOut = [nout * i for i in pruning_ratio]
        nOut = sorted(list(set(nOut)))
        return tuple(nOut)


def ChannelWiseInter(inputs, oC, mode = 'v2'):
    if mode == 'v1':
        return ChannelWiseInterV1(inputs, oC)
    elif mode == 'v2':
        return ChannelWiseInterV2(inputs, oC)
    else:
        raise ValueError('invalid mode = {:}'.format(mode))

def ChannelWiseInterV1(inputs, oC):
    assert inputs.dim() == 4, 'invalid dimension = {:}'.format(inputs.dim())
    def start_index(a, b, c):
        return int(math.floor(float(a * c) / b))
    def end_index(a, b, c):
        return int(math.ceil(float((a + 1) * c) / b))

    batch, iC, H, W = inputs.size()
    outputs = torch.zeros((batch, oC, H, W), dtype = inputs.dtype, device = inputs.device)
    if iC == oC:
        return inputs
    for i in range(oC):
        istartT, iendT = start_index(i, oC, iC), end_index(i, oC, iC)
        values = inputs[:, istartT:iendT].mean(dim = 1)
        outputs[:,i,:,:] = values
    return outputs

def ChannelWiseInterV2(inputs, oC):
    assert inputs.dim() == 4, 'invalid dimension = {:}'.format(inputs.dim())
    batch, C, H, W = inputs.size()
    if C == oC:
        return inputs
    else:
        return nn.functional.adaptive_avg_pool3d(inputs, (oC, H, W))



def Align(inputs, output_channel_index, nout):
    batch, h, w = inputs.size()[0], inputs.size()[2], inputs.size()[3]
    mask = torch.zeros((batch, nout, h, w))
    for num, each_batch in enumerate(inputs):
        for idx, vec in zip(output_channel_index, each_batch):
            mask[each_batch][idx] = vec
    return mask



