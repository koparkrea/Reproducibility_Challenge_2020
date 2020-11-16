import os, sys, pickle, random, argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from shutil import copyfile
from copy import deepcopy

cwd = os.getcwd()
sys.path.append(cwd)
from procedures.search_main import get_procedures
from procedures.optimizers import get_optim_scheduler
from procedures.SearchDatasetWrap import SearchDataset
from procedures.get_dataset_with import get_datasets
from procedures.model_selection import obtain_search_model
from config_utils.configure_utils import load_config
from procedures.flop_benchmark import get_model_info

def obtain_search_args():
    parser = argparse.ArgumentParser(description= 'Train a classification model on typical image classification datasets')
    parser.add_argument('--resume', action = 'store_true', default = False, help='Resume path')
    parser.add_argument('--model_config', type = str, help='the path to the model configuration')
    parser.add_argument('--optim_config', type = str, help='the path to the optimizer configuration')
    parser.add_argument('--split_path', type = str, help='the split file path')
    parser.add_argument('--search_shape', type = str, help='the shape to be searched(depth / width)')
    parser.add_argument('--gumbel_tau_max', type = float, help='the maximum tau')    
    parser.add_argument('--gumbel_tau_min', type = float, help='the minimum tau')
    parser.add_argument('--procedure', type = str, help='the procedure prefix(basic / search)')
    parser.add_argument('--FLOP_ratio', type = float, help='the expected flop ratio')
    parser.add_argument('--FLOP_weight', type = float, help='the loss weight for FLOP')
    parser.add_argument('--FLOP_tolerant', type = float, help='the tolerant range for FLOP')
    parser.add_argument('--batch_size', type = int, default=2, help='batch size for training')
    # Data Generation
    parser.add_argument('--dataset',          type=str,                   help='The dataset name.')
    parser.add_argument('--data_path',        type=str,                   help='The dataset name.')
    parser.add_argument('--cutout_length',    type=int,                   help='The cutout length, negative means not use.')
    # Printing
    parser.add_argument('--print_freq',       type=int,   default=100,    help='print frequency (default: 200)')
    parser.add_argument('--print_freq_eval',  type=int,   default=100,    help='print frequency (default: 200)')
    # Checkpoints
    parser.add_argument('--eval_frequency',   type=int,   default=1,      help='evaluation frequency (default: 200)')
    # Acceleration
    parser.add_argument('--workers',          type=int,   default=8,      help='number of data loading workers (default: 8)')
    # Random Seed
    parser.add_argument('--rand_seed',        type=int,   default=-1,     help='manual seed')
    parser.add_argument('--pretrain', action = 'store_true', default = False, help = 'the path to pretrained model')
    args = parser.parse_args()


    if args.rand_seed is None or args.rand_seed <0:
        args.rand_seed = random.randint(1,100000)
    assert args.gumbel_tau_max is not None and args.gumbel_tau_min is not None, 'tau range cannot be none'
    assert args.FLOP_tolerant is not None and args.FLOP_tolerant > 0, 'invalid FLOP_tolerant = {:}'.format(args.FLOP_tolerant)
    return args

# fix the rand_seed for exact same result
def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(args.workers)
    
    # Fix seed for equivalent result
    prepare_seed(args.rand_seed)

    train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, args.cutout_length)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory = True)


    split_file_path = Path(args.split_path)
    assert split_file_path.exists(), 'invalid path = {:}'.format(split_file_path)
    split_info = torch.load(split_file_path)

    train_split, valid_split = split_info['train'], split_info['valid']
    assert len(set(train_split).intersection(set(valid_split))) == 0, 'there should be no intersection between train/split'
    assert len(train_split) + len(valid_split) == len(train_data)
    
    search_dataset = SearchDataset(args.dataset, train_data, train_split, valid_split) 

    search_valid_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_split), pin_memory=True, num_workers=args.workers)
    search_loader = torch.utils.data.DataLoader(search_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.workers, pin_memory= True, sampler= None)

    # obtain configures
    model_config = load_config(args.model_config, {'class_num' : class_num, 'search_mode' : args.search_shape} ) # NEED to IMPROVE

    # obtain search model
    search_model = obtain_search_model(model_config)


    max_flop, param = get_model_info(search_model, xshape)
    print('max_flop is ',max_flop)
    optim_config = load_config(args.optim_config, {'class_num' : class_num, 'FLOP' : max_flop})
    base_optimizer, scheduler, criterion = get_optim_scheduler(search_model.base_parameters(), optim_config)
    arch_optimizer = torch.optim.Adam(search_model.arch_parameters(), lr = optim_config.arch_LR, betas= (0.5, 0.999), weight_decay= optim_config.arch_decay)


    model_base_path, model_best_path = os.path.join(os.getcwd(),'result','state_dict','search','checkpoint.th'), os.path.join(os.getcwd(),'result','state_dict','search', 'best.th')
    
    #network, criterion =torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
    network, criterion = search_model.cuda(), criterion.cuda()


    # load pretrained state_dict
    if args.pretrain:
        pretrain_path = os.path.join(os.getcwd(),'result','state_dict','basic','best_path','best.th')
        pretrain_checkpoint = torch.load(pretrain_path)
        base_optimizer.load_state_dict(pretrain_checkpoint['optimizer'])
        search_model.load_state_dict(pretrain_checkpoint['search_model'])
        print('------------------------loading pretrained model--------------------------')
    # load checkpoint
    if args.resume:
        resume_path = model_base_path
        checkpoint = torch.load(resume_path)
        print('-'*10 + 'loading resume checkpoint' + '-'*10)        
        start_epoch = checkpoint['epoch'] + 1
        search_model.load_state_dict(checkpoint['search_model'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        base_optimizer.load_state_dict(checkpoint['base_optimizer'])
        arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        valid_accuracies = checkpoint['valid_accuracies']
        arch_genotypes = checkpoint['arch_genotypes']
    else:
        start_epoch, valid_accuracies, arch_genotypes = 0, {'best':-1}, {}
        print('-'*10 + 'initilalize start epoch and valid_accuracies'+'-'*10) 

    # main procedure
    train_func, valid_func = get_procedures(args.procedure)
    total_epoch = optim_config.epochs + optim_config.warmup
    layer_by_ratio = {}
    
    for epoch in range(start_epoch, total_epoch):
        scheduler.update(epoch, 0)
        search_model.set_tau(args.gumbel_tau_max, args.gumbel_tau_min, epoch*1.0/total_epoch)
        epoch_str = 'epoch={:03d}/{:03d}'.format(epoch, total_epoch)
        LRs = scheduler.get_lr()
        find_best = False

        # Path
        cwd = os.getcwd()        
        ratio_path_a = os.path.join(cwd,'result',str(args.dataset) + '_' + str(args.search_shape))
        ratio_path_b = 'gumbel_{:}_{:}_epoch_{:}_{:}_best.pkl'.format(args.gumbel_tau_min,args.gumbel_tau_max,epoch,total_epoch)
        ratio_path_c = 'which_criteria_epoch_{:}_{:}_best.pkl'.format(epoch, total_epoch)
        ratio_path_d = 'which_ratio_epoch_{:}_{:}_best.pkl'.format(epoch, total_epoch)
        ratio_path = os.path.join(ratio_path_a, ratio_path_b)
        ratio_path_ = os.path.join(ratio_path_a, ratio_path_c)
        ratio_path__ = os.path.join(ratio_path_a, ratio_path_d)

        # train for one epoch
        train_arch_loss, train_acc1, train_acc5 = train_func(search_loader, network, criterion, scheduler, base_optimizer, arch_optimizer, optim_config, {'epoch-str':epoch_str, 'FLOP-exp':max_flop * args.FLOP_ratio, 'FLOP-weight':args.FLOP_weight, 'FLOP-tolerant': max_flop * args.FLOP_tolerant}, args.print_freq)
        print('******** TRAIN [{:}] arch-loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}'.format(epoch_str, train_arch_loss, train_acc1, train_acc5))
        cur_FLOP, genotype = search_model.get_flop('genotype',model_config._asdict(), None)
        arch_genotypes[epoch] = genotype
        
        """
        # evaluate performance
        if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
            valid_loss, valid_acc1, valid_acc5 = valid_func(search_valid_loader, network, criterion, epoch_str, args.print_freq_eval)
            valid_accuracies[epoch] = valid_acc1
        print('******** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f} | Best-valid-Acc@1 = {:.2f}, Error@1 = {:.2f}'.format(epoch_str, valid_loss, valid_acc1, valid_acc5, valid_accuracies['best'], 100-valid_accuracies['best']))
        """

        if train_acc1 > valid_accuracies['best']:
            valid_accuracies['best'] = train_acc1
            arch_genotypes['best'] = genotype
            find_best = True
            #print('Currently, the best validation accuracy found at {:03d}-epoch :: acc@1 = {:.2f}, error@1 = {:.2f}, error@5 = {:.2f}'.format(epoch, valid_acc1, valid_acc5, 100-valid_acc1, 100-valid_acc5))
        

        # Save selected pruning ratio and criteria for each layer
        index_list = arch_genotypes[epoch]['xchannels']
        cri_list = arch_genotypes[epoch]['layer_criteria']
        probe_list = arch_genotypes[epoch]['probe_list']
        with open(ratio_path, 'wb') as f:
            pickle.dump(index_list,f)
        with open(ratio_path_, 'wb') as f:
            pickle.dump(cri_list,f)
        with open(ratio_path__,'wb') as f:
            pickle.dump(probe_list,f)

        # save current loader
        save_path = save_checkpoint({
            'epoch' : epoch,
            'args' : deepcopy(args),
            'valid_accuracies' : deepcopy(valid_accuracies),
            'model-config' : model_config._asdict(),
            'optim-config' : optim_config._asdict(),
            'search_model' : search_model.state_dict(),
            'base_optimizer' : base_optimizer.state_dict(),
            'arch_optimizer' : arch_optimizer.state_dict(),
            'arch_genotypes' : arch_genotypes,
            'scheduler' : scheduler.state_dict()
            }, model_base_path)
        print('-'*10 +"saving current epoch's checkpoint" + "-"*10)
        if find_best:
            copy_checkpoint(model_base_path, model_best_path)
            print('-'*10 + 'saving best checkpoint' + '-'*10)
    torch.save({
        'valid_loader' : valid_loader,
        'search_loader' : search_loader,
        }, os.path.join(os.getcwd(), 'data','loader','loader.th'))
    print('-'*10 + 'saving loader' +'-'*10)

def save_checkpoint(state, filename):
    if os.path.exists(filename):
        os.remove(filename)
    torch.save(state, filename)
    assert os.path.isfile(filename), 'save filename : {:} is failed, which is not found'.format(filename)
    return filename

def copy_checkpoint(base, best):
    if os.path.isfile(best):
        os.remove(best)
    copyfile(base, best)



if __name__ == '__main__':
    args = obtain_search_args()
    main(args)



