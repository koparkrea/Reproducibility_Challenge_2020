import pickle, os
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict

def filter_selection(weight, index, criteria):
    output_channel_index = []
    weight_vec = weight.view(weight.size(0),-1)

    if criteria == 'l1-norm':
        norm = torch.norm(weight_vec, 1, dim = 1)
        norm_np = norm.cpu().detach().numpy()
        arg = np.argsort(norm_np)
        arg_index = arg[::-1][:index]
        output_channel_index = sorted(arg_index.tolist())
    elif criteria == 'l2-norm':
        norm = torch.norm(weight_vec, 2, dim = 1)
        norm_np = norm.cpu().detach().numpy()
        arg = np.argsort(norm_np)
        arg_index = arg[::-1][:index]
        output_channel_index = sorted(arg_index.tolist())
    elif criteria == 'l2-GM':
        weight_vec_np = weight_vec.cpu().detach().numpy()
        weight_matrix = distance.cdist(weight_vec_np, weight_vec_np)
        matrix_sum = np.sum(weight_matrix, axis = 0)
        output_channel_index = sorted(np.argpartition(matrix_sum, -index)[-index:].tolist())
    return output_channel_index


def pick_index_cri(path,number = 599):
    index_list = os.path.join(os.getcwd(), 'result', path, 'gumbel_0.1_5.0_epoch_' + str(number) + '_600_best.pkl')
    cri_list = os.path.join(os.getcwd(), 'result', path, 'which_criteria_epoch_' + str(number) + '_600_best.pkl')

    with open(index_list, 'rb') as f:
        index = pickle.load(f)
    with open(cri_list, 'rb') as f:
        criteria = pickle.load(f)
    return index, criteria

def get_pruned_model(model, path, number):
    pretrained_model = os.path.join(os.getcwd(), 'result', 'state_dict', 'basic', 'best_path', 'best.th')
    checkpoint = torch.load(pretrained_model)
    model_weight = checkpoint['search_model'].copy()
    pruned_weight = checkpoint['search_model'].copy()
    index, criteria = pick_index_cri(path, number)

    del model_weight['width_attention']
    del model_weight['criteria_attention']
    del pruned_weight['width_attention']
    del pruned_weight['criteria_attention']
 
    output_channel_index = filter_selection(model_weight['layers.0.weights'], index[1], criteria[0])
    del pruned_weight['layers.0.weights']
    #pruned_weight['layers.0.weights'] = model_weight['layers.0.weights'][output_channel_index]
    pruned_weight['layers.0.conv.weight'] = model_weight['layers.0.conv.weight'][output_channel_index]
    input_channel_index = 0
    i = 0
    downsample = []
    for c in model_weight:
        if 'downsample' not in c:
            
            if 'layers.0.weights' == c:
                continue
            elif 'layers.0.conv.weight' ==c:
                continue
            elif 'choices_tensor' in c:
                del pruned_weight[c]
                start = i
                i +=1

            elif 'conv' and 'weights' in c:
                output_channel_index = filter_selection(model_weight[c], index[i+1], criteria[i])
                del pruned_weight[c]
                continue
                """
                print(len(index), len(criteria),i)
                output_channel_index = filter_selection(model_weight[c], index[i+1], criteria[i])
                input_weight= model_weight[c][:, input_channel_index]
                pruned_weight[c] = input_weight[output_channel_index]
                print(c)
                print(i)
                """
            elif 'conv.weight' in c:
                input_weight = model_weight[c][:, input_channel_index]
                pruned_weight[c] = input_weight[output_channel_index]
                if 'conv_a' in c:
                    input_channel_index_d = input_channel_index

            elif 'classifier' in c:
                if 'weight' in c:
                    pruned_weight[c] = model_weight[c][:,output_channel_index]
                else:
                    continue
            elif 'num_batches_tracked' in c:
                pruned_weight[c] = torch.Tensor([0])
                input_channel_index = output_channel_index
            else:
                pruned_weight[c] = model_weight[c][output_channel_index]
            
        elif 'downsample' in c:
            if 'downsample.weights' in c:
                output_channel_index_d = filter_selection(model_weight[c], index[start+1], criteria[start])
                del pruned_weight[c]
                """
                input_weight = model_weight[c][:,input_channel_index_d]
                print(input_weight.shape, len(output_channel_index_d))
                pruned_weight[c] = input_weight[output_channel_index_d]
                print(c)
                """
            elif 'downsample.choices_tensor' in c:
                del pruned_weight[c]

            elif 'downsample.conv.weight' in c:
                #output_channel_index = filter_selection(model_weight[c], index[downsample[start]+1], criteria[downsample[start]])
                input_weight = model_weight[c][:,input_channel_index_d]
                pruned_weight[c] = input_weight[output_channel_index_d]

            elif 'downsample.BNs.num_batches_tracked' in c:
                pruned_weight[c] = torch.Tensor([0])

            else:
                pruned_weight[c] = model_weight[c][output_channel_index]

        else:
            continue


    """
    start = 0

    for c in model_weight:
        if 'downsample' in c:
            if 'conv' and 'weights' in c:
                output_channel_index = filter_selection(model_weight[c], index[downsample[start]+1], criteria[downsample[start]])
                input_weight = model_weight[c][:input_channel_index_d]
                pruned_weight[c] = model_weight[c][output_channel_index]

            elif 'conv.weight' in c:
                output_channel_index = filter_selection(model_weight[c], index[downsample[start]+1], criteria[downsample[start]])
                input_weight = model_weight[c][:input_channel_index_d]
                pruned_weight[c] = model_weight[c][output_channel_index]

            elif 'choices_tensor' in c:
                del pruned_weigth[c]

                
            elif 'downsample.BNs.num_batches_tracked' in c:
                pruned_weight[c] = torch.Tensor([0])
                input_channel_index_d = output_channel_index
            
        else:
            continue
    """

    model.load_state_dict(pruned_weight, strict=False )
    return model



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
            if (not torch.isinf(gumbels).any()) and (not torch.isinf(probs).any()) and (not torch.isnan(probs).any()):
                break

    return probs


def selectwithP(parameter, tau, just_prob = False, num = 2, eps = 1e-7):
    if tau <= 0:
        new_logits = parameter
        probs = nn.functional.softmax(parameter, dim = 1)
    else:
        while True:
            gumbels = - torch.empty_like(parameter).exponential_().log()
            new_logits = (parameter.log_softmax(dim = 1) + gumbels) / tau
            probs = nn.functional.softmax(new_logits, dim = 1)
            if (not torch.isinf(gumbels).any()) and (not torch.isinf(probs).any()) and (not torch.isnan(probs).any()):
                break
    if just_prob:
        return probs

    with torch.no_grad():
        probs = probs.cpu()
        selected_index = torch.multinomial(probs + eps, num, False).to(parameter.device)
    selected_param = torch.gather(new_logits, 1, selected_index)
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
        nOut = [int(nout * i) for i in pruning_ratio]
        nOut = sorted(list(set(nOut)))
        return tuple(nOut)

def get_width_choice_finetune(stage):
    pruning_ratio = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if stage == 0:
        nout = 16
    elif stage == 1:
        nout = 32
    elif stage == 2:
        nout = 64
    nOut = [int(nout*i) for i in pruning_ratio]
    nOut = sorted(list(set(nOut)))
    return tuple(nOut)

def get_width_choice_limit(nout, stage):
    if stage == 0 or stage == 1:
        pruning_ratio = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif stage == 2:
        pruning_ratio = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        raise ValueError('invalid stage : {:}'.format(stage))

    if nout is None:
        return len(pruning_ratio)
    else:
        nOut = [int(nout*i) for i in pruning_ratio]
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
    batch, h, w = inputs.shape[0], inputs.shape[2], inputs.shape[3]
    mask = torch.zeros((batch, nout, h, w),device = inputs.device)
    #mask = np.zeros((batch,nout,h,w))
    """
    for num, each_ba in enumerate(inputs):
        for idx, vec in zip(output_channel_index, each_ba):
            mask[num][idx] = vec
    return mask
    """
    mask[:,output_channel_index] = inputs
    return mask
