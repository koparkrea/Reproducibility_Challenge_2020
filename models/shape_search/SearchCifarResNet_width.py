import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance

from utils import get_width_choice, addictive_func
from utils import selectwithP, linear_forward, ChannelWiseInter


def criteria_selection(weights, index, criteria_):
    output_channel_index = []
    weight_vec = weights.view(weights.size()[0], -1)


    if criteria_ == 'l1_norm':
        norm = torch.norm(weight_vec, 1, dim = 1)
        norm_np = norm.cpu().detach().numpy()
        arg_max = np.argsort(norm_np)
        arg_max_index = arg_max[::-1][:index]
        output_channel_index = sorted(arg_max_index.tolist())

    elif criteria_ == 'l2_norm':
        norm = torch.norm(weight_vec, 2, dim = 1)
        norm_np = norm.cpu().detach().numpy()
        arg_max = np.argsort(norm_np)
        arg_max_index = arg_max[::-1][:index]
        output_channel_index = sorted(arg_max_index.tolist())

    elif criteria_ == 'l2_GM':
        weight_vec_np = weight_vec.cpu().detach().numpy()
        matrix = distance.cdist(weight_vec_np, weight_vec_np, metric= 'euclidean')
        sum_ = np.sum(np.abs(matrix), axis = 0)
        output_channel_index = np.argpartition(sum_, -index)[-index:]
    
    else:
        raise ValueError('invalid criteria = {:}'.format(criteria_))

    return output_channel_index

# criteria forward
def criteria_forward(inputs, conv, weights, index, criteria_set_list):
    idx = []
    for criteria_ in criteria_set_list:
        cri_index = criteria_selection(weights, index, criteria_)
        idx.append(cri_index)
        out = conv(inputs)
    selected = [out[:,oC] for oC in idx]
    return selected, idx

# Setup basic conv layer forward
class ConvBNRelu(nn.Module):
    num_conv = 1
    def __init__(self, nin, nout, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNRelu,self).__init__()
        self.Inshape = None
        self.Outshape = None
        self.choices = get_width_choice(nout)
        self.register_buffer('choices_tensor', torch.Tensor(self.choices))

        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        else:
            self.avg = None
        
        self.conv = nn.Conv2d(nin, nout, kernel_size = kernel, stride = stride, padding = padding, dilation = 1 , groups = 1, bias = bias)

        if has_bn:
            self.BNs = nn.ModuleList()
        for i, _out in enumerate(self.choices):
            self.BNs.append(nn.BatchNorm2d(_out))

        if has_relu:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.relu = None
        self.weights = self.conv.weight
        self.in_dim = nin
        self.out_dim = nout
        self.search_mode = 'basic'

    def get_range(self):
        return [self.choices]

    def get_flops(self, channels, check_range = True, divide = 1):
        iC, oC = channels
        if check_range:
            assert iC <= self.conv.in_channels and oC <= self.conv.out_channels
        assert isinstance(self.Inshape, tuple) and len(self.Inshape) == 2, 'invalid input H,W = {:}'.format(self.Inshape)
        assert isinstance(self.Outshape, tuple) and len(self.Outshape) == 2, 'invalid output H,W = {:}'.format(self.Outshape)
        conv_per_position_flops = (self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups)
        all_positions = self.Outshape[0] * self.Outshape[1]
        flops = (conv_per_position_flops * all_positions / divide) * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops



    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search mode = {:}'.format(self.search_mode))


    def basic_forward(self, inputs):
        if self.has_avg:
            out = self.avg(inputs)
        else:
            out = inputs
        
        conv = self.conv(out)
        
        if self.BNs:
            out = self.BNs[-1](conv)
        else:
            out = conv    

        if self.has_relu:
            out = self.relu(out)
        else:
            out = out

        if self.Inshape is None:
            self.Inshape = (inputs.size(-2), inputs.size(-1))
            self.Outshape = (out.size(-2), out.size(-1))
        return out

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple, tuple_inputs) and len(tuple_inputs) == 7, 'invalid inputs'
        inputs, expected_inC, probability, index, prob, criteria_set, criteria_set_prob = tuple_inputs

        index, prob = torch.squeeze(index).tolist(), torch.squeeze(prob)
        probability = torch.squeeze(probability)
        assert len(index) == 2, 'invalid length of index = {:}'.format(index)

        # compute expected flop
        expected_outC = (self.choices_tensor * probability).sum()
        expected_flop = self.get_flops([expected_inC, expected_outC], False, 1e6)

        if self.has_avg:
            out = self.avg(inputs)
        else:
            out = inputs

        out_conv_list = []
        out_criteria = []
        for i in range(index):
            out_conv, output_channel_index = self.criteria_forward(out, self.conv, self.weights, i, [cri_set for cri_set in criteria_set])
            ######## NEED TO ADJUST BN LAYER !!!!!!
            out_bn = [self.BNs[idx](_conv) for idx,_conv in zip(index, out_conv)] 

            out_conv_list.append(out_bn)
            ######## NEED TO ADD ALIGN FUNCTION!!!!!
            out_align_list = []
            for set_ in range(len(criteria_set)):
                out_align = Align(out_conv_list[set_], output_channel_index, self.out_dim) * criteria_set_prob[set_]
                out_align_list.append(out_align)
            out_criteria_ = sum(out_align_list)
            out_criteria.append(out_criteria_)

        # merge by weighted sum
        out = out_criteria[0] * prob[0] + out_criteria[1] * prob[1]

        if has_relu:
            out = self.relu(out)
        else:
            out = out
        return out, expected_outC, expected_flop    
        

class ResNetBasicblock(nn.Module):
    expansion = 1
    num_conv = 2
    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNRelu(inplanes, planes, 3, stride, 1, False, True, True, True)
        self.conv_b = ConvBNRelu(planes, planes, 3, 1 ,1, False, True, True, False)

        # downsample
        if stride != 1:
            self.downsample = ConvBNRelu(inplanes, planes, 1, 1, 0, False, True, False, False)
        elif inplanes != planes:
            self.downsample = ConvBNRelu(inplanes, planes, 1, 1, 0 , False, False, True, False) # Can't UNDERSTAND WHY BN HERE
        else:
            self.downsample = None
        self.search_mode = 'basic'
        self.out_dim = planes

    def get_range(self):
        return self.conv_a.get_range() + self.conv_b.get_range()

    def get_flops(self, channels):
        assert len(channels) == 3, 'invalid channels = {:}'.format(channels)
        flop_a = self.conv_a.get_flops([channels[0], channels[1]])
        flop_b = self.conv_b.get_flops([channels[1], channels[2]])
        if hasattr(self.downsample, 'get_flops'):
            flop_c = self.downsample.get_flops([channels[0], channels[2]])
        else:
            flop_c = 0
        
        return flop_a + flop_b + flop_c

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search mode = {:}'.format(self.search_mode))

    def basic_forward(inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        else:
            identity = inputs
        out = addictive_func(identity,basicblock)
        return nn.functional.relu(out, inplace = True)

    def search_forward(tuple_inputs):
        assert isinstance(tuple, tuple_inputs) and lens(tuple_inputs) == 7, 'invalid input = {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, index, prob, criteria_set, criteria_set_prob = tuple_inputs
        assert len(index) == 2 and prob.size(0) == 2 and probability.size(0) == 2

        out_a, expected_inC_a, expected_flop_a = self.conv_a((inputs, expected_inC, probability[0], index[0], prob[0],criteria_set, criteria_set_prob[0]))
        out_b, expected_inC_b, expected_flop_b = self.conv_b((out_a, expected_inC_a, probability[1], index[1], prob[1], criteria_set, criteria_set_prob[1]))

        if self.downsample is not None:
            identity, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[1], index[1], prob[1], criteria_set, criteria_set_prob[1]))
        else:
            identity, expected_flop_c = inputs, 0

        out = addictive_func(identity, out_b)
        return nn.functional.relu(out, inplace = True)

class ResNetBottleneck(nn.Module):
    num_conv = 3
    expansion = 4
    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        self.conv_1x1 = ConvBNRelu(inplanes, planes, 1, 1, 0, False, False, True, True)
        self.conv_3x3 = ConvBNRelu(planes, planes, 3, stride, 1, False, False, True, True)
        self.conv_1x4 = ConvBNRelu(planes, planes*expansion, 1, 1, 0, False, False, True, False)

        if stride == 2:
            self.downsample = ConvBNRelu(inplanes, plane*expansion, 1,1,0, False, True, False, False)
        elif inplanes != plane*expansion:
            self.downsample = ConvBNRelu(inplanes, plane*expansion, 1,1,0, False, False, True, False)
        else:
            self.downsample = None
    
        self.out_dim = planes * expansion
        self.search_mode = 'basic'
    
    def get_range(self):
        return self.conv_1x1.get_range() + self.conv_3x3.get_range() + self.conv_1x4.get_range()

    def get_flops(self, channels):
        assert len(channels) == 4, 'invalid channels = {:}'.format(channels)
        flop_a = self.conv_1x1.get_flops([channels[0], channels[1]])
        flop_b = self.conv_3x3.get_flops([channels[1], channels[2]])
        flop_c = self.conv_1x4.get_flops([channels[2], channels[3]])
        if hasattr(self.downsample, 'get_flops'):
            flop_d = self.downsample.get_flops([channels[0], channels[3]])
        else:
            flop_d = 0
        
        return flop_a + flop_b + flop_c + flop_d

    
    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def basic_forward(self, inputs):
        residual = self.conv_1x1(inputs)
        residual = self.conv_3x3(residual)
        residual = self.conv_1x4(residual)
        if self.downsample is not None:
            identity = self.downsample(inputs)
        else:
            identity = inputs
        out = addictive_func(identity, residual)
        return nn.functional.relu(out, inplace = True)

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple, tuple_inputs) and len(tuple_inputs) == 7,'invalid input = {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs, criteria_set, criteria_set_prob = tuple_inputs
        assert indexes.size(0) == 3 and probs.size(0) == 3 and probability.size(0) == 3
        
        out_1x1 , expected_inC_1x1, expected_flop_1x1 = self.conv_1x1((inputs, expected_inC, probability[0], indexes[0], probs[0], criteria_set, criteria_set_prob[0]))
        out_3x3 , expected_inC_3x3, expected_flop_3x3 = self.conv_3x3((out_1x1, expected_inC_1x1, probability[1], indexes[1], probs[1], criteria_set, criteria_set_prob[1]))
        out_1x4 , expected_inC_1x4, expected_flop_1x4 = self.conv_1x4((out_3x3, expected_inC_3x3, probability[2], indexes[2], probs[2], criteria_set, criteria_set_prob[2]))

        if self.downsample is not None:
            identity, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[2], indexes[2], probs[2], criteria_set, criteria_set_prob[2]))
        else:
            identity, expected_flop_c = inputs, 0
        
        out = addictive_func(identity, out_1x4)
        return nn.functional.relu(out, inplace = True), expected_inC_1x4, sum([expected_flop_1x1, expected_flop_1x4, expected_flop_3x3, expected_flop_c])

class SearchWidthCifarResNet(nn.Module):
    def __init__(self, block_name, depth, num_classes, criteria_set):
        super(SearchWidthCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) / 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) / 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) / 9 == 0 , 'depth should be one of 164'
            layer_blocks = (depth - 2) / 9
        else:
            raise ValueError('invalid blockname(ResNetBasicblock / ResNetBottleneck) = {:}'.format(block_name))
        
        self.criteria_set = criteria_set
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNRelu(3, 16, 3, 1, 1, False, False, True, True)])
        self.Inshape = None

        for stage in range(3):
            for i in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * (2**stage)
                stride = 2 if stage > 0 and i == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
        
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.Inshape = None
        self.tau = -1
        self.search_mode = 'basic'

        # set parameters for depth search
        # need to setup parameters FOR CRITERION SEARCH
        self.Ranges = []
        self.layer2indexRange = []
        for idx, layer in enumerate(self.layers):
            start_index = len(self.Ranges)           
            self.Ranges += layer.get_range()
            self.layer2indexRange.append((start_index, len(self.Ranges)))
        assert len(self.Ranges) + 1 == depth, 'invalid depth check = {:}'.format(len(self.Ranges), depth) 

        self.register_parameter('width_attention', nn.Parameter(torch.Tensor(len(self.Ranges), get_width_choice(None))))
        self.register_parameter('criteria_attention', nn.Parameter(torch.Tensor(len(self.Ranges), 3)))        
        nn.init.normal_(self.width_attention, 0, 0.01)
        nn.inti.normal_(self.criteria_attention, 0, 0.01)
        self.apply(initialize_resnet)   

    def arch_parameters(self):
        return [self.width_attention]

    def base_parameters(self):
        return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())


    def get_flops(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        channels = [3]
        for i, weight in enumerate(self.width_attentions):
            if mode == 'genotype':
                with torch.no_grad():
                    probe = nn.functional.softmax(weight, dim = 0)
                    c = self.Ranges[i][torch.argmax(probe).item()]
            else:
                raise ValueError('invalid mode = {:}'.format(mode))
            channels.append(c)
        flop = 0
        for i, layer in enumerate(self.layers):
            s,e = self.layer2indexRange[i]
            chls = tuple(channels[s:e+1])
            flop += layer.get_flops(chls)
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1e6
        else:
            config_dict['xchannels'] = channels
            config_dict['super_type'] = 'infer-width'
            config_dict['estimated_FLOP'] = flop / 1e6
            return flop / 1e6, config_dict


    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch ratio = {:}'.format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2 
        self.tau = tau

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search mode = {:}'.format(self.search_mode))
    
    def basic_forward(self, inputs):
        if self.Inshape is None:
            self.Inshape = (inputs.size(-2), inputs.size(-1))
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits

    def search_forward(self, inputs):
        ratio_probs = nn.functional.softmax(self.width_attention, dim = 1)
        selected_widths, selected_probs = selectwithP(self.width_attention, self.tau)
        selected_criteria_probs = selectwithP_criteria(self.criteria_attention, self.tau)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()
        
        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        for i, layer in enumerate(self.layers):
            selected_widths_l = selected_widths[last_channel_idx:last_channel_idx + layer.num_conv]
            selected_probs_l = selected_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            
            # for criteria selection probs
            selected_criteria_probs_l = selected_criteria_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            
            eachlayer_entire_probs = ratio_probs[last_channel_idx:last_channel_idx + layer.num_conv]

            x, expected_inC, expected_flop = layer((x, expected_inC, eachlayer_entire_probs, selected_widths_l, selected_probs_l, self.criteria_set, selected_criteria_probs_l))
            last_channel_idx += layer.num_conv
            flops.append(expected_flop)
        flops.append(expected_inC * (self.classifier.out_features*1.0/1e6))
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        logits = linear_forward(out, self.classifier)
        return logits, torch.stack([sum(flops)])


 
