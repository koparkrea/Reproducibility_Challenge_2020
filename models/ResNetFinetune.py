import torch, os, pickle, sys
import numpy as np
import torch.nn as nn

cwd = os.getcwd()
sys.path.append(cwd)
from procedures.BNselection import BNselection
from procedures.utils_criteria import get_width_choice_finetune, addictive_func

class ConvBNRelu(nn.Module):
    def __init__(self, inplane, outplane, kernel_size, stride, padding, bias, has_avg, has_bn, has_relu, stage):
        super(ConvBNRelu, self).__init__()
        
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(inplane, outplane, kernel_size = kernel_size, stride= stride, padding= padding, dilation=1, groups=1, bias=bias)
        self.has_bn = has_bn
        #self.BNs = nn.ModuleList()
        #self.choices = get_width_choice_finetune(stage)
        #self.register_buffer('choices_tensor', torch.Tensor(self.choices))
        self.BNs = nn.BatchNorm2d(outplane)

        if has_relu:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.relu = None
        
    def forward(self, inputs, index_tuple):
        assert isinstance(index_tuple, tuple) and len(index_tuple) == 2
        index_ratio, index_cri = index_tuple
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.has_bn:
            out = self.BNs(conv)
        else:
            out = conv

        if self.relu:
            out = self.relu(out)
        else:
            out = out
        
        return out



class ResNetBasicblock(nn.Module):
    num_conv = 2
    def __init__(self, list_channel, stride, stage):
        super(ResNetBasicblock, self).__init__()
        assert len(list_channel) == 3, 'invalid input channel : {:}'.format(len(list_channel))
        inplane, middle_plane, out_plane = list_channel
        assert stride == 1 or stride == 2, 'invalid stride : {:}'.format(stride)
        self.conv_a = ConvBNRelu(inplane, middle_plane, 3, stride, 1, False, False, True, True, stage)
        self.conv_b = ConvBNRelu(middle_plane, out_plane, 3, 1, 1, False, False, True, False, stage)

        # downsample
        if stride == 2:
            self.downsample = ConvBNRelu(inplane, out_plane, 1, 1, 0, False, True, False, False, stage)
        elif inplane != out_plane:
            self.downsample = ConvBNRelu(inplane, out_plane, 1, 1, 0, False, False, True, False, stage)
        else:
            self.downsample = None
        self.out_dim = out_plane

    def forward(self, inputs, index_tuple):
        assert isinstance(index_tuple, tuple) and len(index_tuple) == 2
        index_ratio, index_cri = index_tuple
        x = inputs
        x = self.conv_a(x, (index_ratio[0], index_cri[0]))
        x = self.conv_b(x, (index_ratio[1], index_cri[1]))

        if self.downsample is not None:
            identity = self.downsample(inputs, (index_ratio[1], index_cri[1]))
        else:
            identity = inputs
        out = addictive_func(x, identity)
        return nn.functional.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    num_conv = 3
    def __init__(self, list_channel, stride, stage):
        super(ResNetBottleneck, self).__init__()
        assert len(list_channel) == 4, 'invalid channel list : {:}'.format(len(list_channel))
        inplane, middle_plane, middle_plane2, out_plane = list_channel
        assert stride == 1 or stride == 2, 'invalid stride : {:}'.format(stride)
        self.conv1x1 = ConvBNRelu(inplane, middle_plane, 1, 1, 0, False, False, True, True, stage)
        self.conv3x3 = ConvBNRelu(middle_plane, middle_plane2, 3, stride, 1, False, False, True, True, stage)
        self.conv1x4 = ConvBNRelu(middle_plane2, out_plane, 1, 1, 0, False, False, True, False, stage)
        # downsample
        if stride == 2:
            self.downsample = ConvBNRelu(inplane, out_plane, 1, 1, 0, False, True, False, False, stage)
        elif inplane != out_plane:
            self.downsample = ConvBNRelu(inplane, out_plane, 1, 1, 0, False, False, True, False, stage)
        
        self.out_dim = out_plane

    def forward(self, inputs, index_tuple):
        assert isinstance(index_tuple, tuple) and len(index_tuple) == 2, 'invalid index tuple : {:}'.format(len(index_tuple))
        index_ratio, index_cri = index_tuple
        x = self.conv1x1(inputs, (index_ratio[0], index_cri[0]))
        x = self.conv3x3(x,(index_ratio[1], index_cri[1]))
        x = self.conv1x4(x, (index_ratio[2], index_cri[2]))

        if self.downsample is not None:
            identity = self.downsample(inputs, (index_ratio[2], index_cri[2]))
        else:
            identity = inputs
        
        out = addictive_fun(x, identity)
        return nn.functional.relu(out, inplace=True)


class CifarResNetFinetune(nn.Module):
    def __init__(self, block_name, depth, num_classes, channel_list):
        super(CifarResNetFinetune, self).__init__()
        if block_name == 'ResNetBasicblock':
            assert (depth -2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110' 
            layer_block = int((depth - 2) / 6)
            block = ResNetBasicblock
        elif block_name == 'ResNetBottleneck':
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_block = int((depth - 2) / 9)
            block = ResNetBottleneck
        else:
            raise ValueError('invalid block_name : {:}'.format(block_name))
        self.num_classes = num_classes
        self.channel_list = channel_list
        self.layers = nn.ModuleList([ConvBNRelu(3, channel_list[1], 3, 1, 1, False, False, True, True, 0)])
        start_index = 0
        self.index_ratios, self.index_cris = BNselection(start_index)
        for stage in range(3):
            for i in range(layer_block):
                stride = 2 if stage > 0 and i == 0 else 1
                if block_name == 'ResNetBasicblock':
                    index_ratio, index_cri = BNselection(start_index+1, 2)
                    module = block(self.channel_list[start_index+1:start_index + 4], stride, stage)
                elif block_name == 'ResNetBottleneck':
                    index_ratio, index_cri = BNselection(start_index+1, 3)
                    module = block(self.channel_list[start_index+1:start_index + 5], stride, stage)
                else:
                    raise ValueError('invalid block_name : {:}'.format(block_name))
                self.layers.append(module)
                self.index_ratios.append(index_ratio)
                self.index_cris.append(index_cri)
                start_index += module.num_conv
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)


    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            index_ratio = self.index_ratios[i]
            index_cri = self.index_cris[i]
            x = layer(x,(index_ratio, index_cri))
            
        out = self.avgpool(x)
        out = out.view(out.shape[0], -1)
        logits = self.classifier(out)
        return out, logits
