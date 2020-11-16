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
from procedures.utils_criteria import pick_index_cri, get_pruned_model


def obtain_search_args():
    parser = argparse.ArgumentParser(description= 'Train a classification model on typical image classification datasets')
    parser.add_argument('--resume', action = 'store_true', default = False, help='Resume path')
    parser.add_argument('--model_config', type = str, help='the path to the model configuration')
    parser.add_argument('--optim_config', type = str, help='the path to the optimizer configuration')
    parser.add_argument('--search_shape', type = str, help='the shape to be searched')
    parser.add_argument('--procedure', type = str, help='the procedure prefix(basic / search)')
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
    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed <0:
        args.rand_seed = random.randint(1,100000)
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

    prepare_seed(args.rand_seed)
    #logger = 
    

    # obtain datasets
    loader = torch.load(os.path.join(os.getcwd(), 'data', 'loader','loader.th'))
    train_loader, valid_loader = loader['search_loader'], loader['valid_loader']
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
	    class_num = 100
    elif args.dataset == 'imagenet_1k':
	    class_num = 1000
    else:
	    raise ValueError('invalid dataset : {:}'.format(args.dataset))
	

    # obtain configures
    model_config = load_config(args.model_config, {'class_num' : class_num, 'search_mode':args.search_shape}) 


    # obtain search model
    search_model = obtain_search_model(model_config)
    search_model = get_pruned_model(search_model, 'cifar10_tolerance0.01_beforealign', 599)
    print('-'*10 + 'loading pruned model' + '-'*10)
    optim_config = load_config(args.optim_config, {'class_num' : class_num})
    optimizer, scheduler, criterion = get_optim_scheduler(search_model.parameters(), optim_config)
    network, criterion = search_model.cuda(), criterion.cuda()


    model_base_path, model_best_path = os.path.join(os.getcwd(),'result','finetune','checkpoint.th'), os.path.join(os.getcwd(),'result','finetune','best.th')


    # load checkpoint
    if args.resume:
        resume_path = model_base_path
        checkpoint = torch.load(resume_path)
        print('-'*10 + 'loading checkpoint' +'-'*10) 
        start_epoch = checkpoint['epoch']
        search_model.load_state_dict(checkpoint['search_model'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        valid_accuracies = checkpoint['valid_accuracies']
    else:
        #best_path = os.path.join(os.getcwd(), 'result','state_dict', 'search', 'best_path', 'best.th')
        #bestpoint = torch.load(best_path)
        #search_model.load_state_dict(bestpoint['search_model'])
        #print('-'*10 + 'loading best model from search_model' + '-'*10)
        start_epoch, valid_accuracies = 0, {'best':-1}
        print('-'*10 + 'initialize start epoch and valid_accuracies' + '-'*10) 

    # main procedure
    train_func, valid_func = get_procedures(args.procedure)
    total_epoch = optim_config.epochs + optim_config.warmup
    
    for epoch in range(start_epoch, total_epoch):
        scheduler.update(epoch, 0)
        epoch_str = 'epoch={:03d}/{:03d}'.format(epoch, total_epoch)
        LRs = scheduler.get_lr()
        find_best = False

        # train for one epoch
        train_loss, train_acc1, train_acc5 = train_func(train_loader, network, criterion, scheduler, optimizer, optim_config, epoch_str, args.print_freq)
        print('******** TRAIN [{:}] base-loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}'.format(epoch_str, train_loss,train_acc1, train_acc5))

        # evaluate performance
        if (epoch % args.eval_frequency == 0) or (epoch + 1 == total_epoch):
            valid_loss, valid_acc1, valid_acc5 = valid_func(valid_loader, network, criterion, optim_config, epoch_str, args.print_freq_eval)
            valid_accuracies[epoch] = valid_acc1
            if valid_acc1 > valid_accuracies['best']:
                find_best = True
                valid_accuracies['best'] = valid_acc1
        print('_'*10 + 'saving checkpoint' + '-'*10) 
        save_path = save_checkpoint({
            'epoch' : epoch,
            'args' : deepcopy(args),
            'valid_accuracies' : deepcopy(valid_accuracies),
            'model-config' : model_config._asdict(),
            'optim-config' : optim_config._asdict(),
            'search_model' : search_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            }, model_base_path)
        if find_best:
            copy_checkpoint(model_base_path, model_best_path)
            print('_'*10 + 'saving best checkpoint' + '-'*10) 


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

