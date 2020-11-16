import sys, os, pickle
cwd = os.getcwd()
sys.path.append(cwd+'/models/')


def obtain_search_model(config):
    print(config.search_mode)
    if config.dataset == 'cifar':
        if config.arch == 'resnet':
            if config.search_mode == 'width':
                from SearchCifarResNet_ratio import SearchWidthCifarResNet
                return SearchWidthCifarResNet(config.module, config.depth, config.class_num)
            elif config.search_mode == 'criteria':
                from SearchCifarResNet_criteria import SearchWidthCifarResNet
                return SearchWidthCifarResNet(config.module, config.depth, config.class_num)
            elif config.search_mode == 'finetune':
                index_path = os.path.join(os.getcwd(),'result', 'cifar10_tolerance0.01_beforealign', 'gumbel_0.1_5.0_epoch_599_600_best.pkl')
                with open(index_path, 'rb') as f:
                    channel_list = pickle.load(f)
                from ResNetFinetune import CifarResNetFinetune
                return CifarResNetFinetune(config.module, config.depth, config.class_num, channel_list)
            elif config.search_mode == 'limit':
                from SearchCifarResNet_criteria_dif import SearchWidthCifarResNet
                return SearchWidthCifarResNet(config.module, config.depth, config.class_num)
            elif config.search_mode == 'criteria_I2':
                from SearchCifarResNet_criteria_1 import SearchWidthCifarResNet
                return SearchWidthCifarResNet(config.module, config.depth, config.class_num)
            else:
                raise ValueError('invalid search mode = {:}'.format(config.search_mode))
        else:
            raise ValueError('invalid arch = {:}'.format(config.arch))
    
    else:
        raise ValueError('invalid dataset = {:}'.format(config.dataset))
