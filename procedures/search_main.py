import os, torch
import sys
cwd = os.getcwd()
sys.path.append(cwd)


from procedures.compute_average import AverageMeter
#Setting 
#torch.cuda.set_device(2)

def obtain_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size()[0]
    _, pred = output.topk(maxk, dim = 1)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    acc_list = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
        acc_list.append(correct_k.mul_(100 / batch_size))
    return acc_list


def get_procedures(procedure):
    procedure = str(procedure)
    train_func = {'basic' : basic_train, 'search' : search_train, 'finetune' : finetune_train, 'scratch' : scratch_train, 'rev':rev_train }
    valid_func = {'basic' : basic_valid, 'search' : search_valid, 'finetune' : finetune_valid, 'scratch' : scratch_valid, 'rev':rev_valid}
    train_func = train_func[procedure]
    valid_func = valid_func[procedure]
    return train_func, valid_func


def change_key(key, value):
    def func(m):
        if hasattr(m, key):
            setattr(m, key, value)
    return func


def get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant):
    expected_flop = torch.mean(expected_flop)

    if flop_cur < flop_need - flop_tolerant:
        loss = - torch.log(expected_flop)
    elif flop_cur > flop_need:
        loss = torch.log(expected_flop)
    else:
        loss = None
    
    if loss is None:
        return 0,0
    else:
        return loss, loss.item()


def search_train(search_loader, network, criterion, scheduler, base_optimizer, arch_optimizer,optim_config, extra_info, print_freq):
    
    arch_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_cls_losses, arch_flop_losses = AverageMeter(), AverageMeter()
    epoch_str, flop_need, flop_weight, flop_tolerant = extra_info['epoch-str'], extra_info['FLOP-exp'], extra_info['FLOP-weight'], extra_info['FLOP-tolerant']

    # train
    network.train()
    print('[Search] : {:}, FLOP-Require={:.2f}, FLOP-Weight={:.2f}'.format(epoch_str, flop_need, flop_weight))
    network.apply(change_key('search_mode', 'search'))
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):
        #scheduler.update(None, 1*step / len(search_loader))
        #base_targets = base_targets.cuda(non_blocking = True)
        arch_inputs = arch_inputs.cuda(non_blocking = True)
        arch_targets = arch_targets.cuda(non_blocking = True)
        """
        # update base parameters
        if pretrain:
            network.eval()
            with torch.no_grad():
                network.apply(change_key('search_mode','basic'))
                logits, expected_flop = network(base_inputs)
            base_loss = criterion(logits, base_targets)
        else:
            base_optimizer.zero_grad()
            logits, expected_flop = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()
            base_optimizer.step()
        """
        # update arch parameters
        arch_optimizer.zero_grad()
        logits, expected_flop = network(arch_inputs)
        flop_cur = network.get_flop('genotype', None, None)
        #flop_cur = network.module.get_flop('genotype', None, None)
        flop_loss, flop_loss_scale = get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant)
        loss = criterion(logits, arch_targets)
        arch_loss = loss + flop_loss * flop_weight
        arch_loss.backward()
        arch_optimizer.step()
        
        # record
        prec1, prec5 = obtain_accuracy(logits.data, arch_targets.data, topk = (1,5))
        top1.update(prec1.item(), arch_inputs.size()[0])
        top5.update(prec5.item(), arch_inputs.size()[0])
        arch_losses.update(arch_loss.item(), arch_inputs.size()[0])
        arch_flop_losses.update(flop_loss_scale, arch_inputs.size()[0])
        arch_cls_losses.update(loss.item(), arch_inputs.size()[0])
    
    print('**TRAIN** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f}\
            Arch-loss:{archloss:.3f}, arch_flop_loss:{archflop:.3f} arch_cls_loss:{clsloss:.3f}'.format(top1=top1, top5=top5, error1 = 100-top1.avg, error5 = 100-top5.avg,\
            archloss=arch_losses.avg, archflop=arch_flop_losses.avg, clsloss=arch_cls_losses.avg))
    print('Current FLOP at {:} is {:}'.format(epoch_str, flop_cur))
    return arch_losses.avg, top1.avg, top5.avg



def search_valid(valid_loader, network, criterion, extra_info, print_freq):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    network.eval()
    network.apply(change_key('search_mode', 'search'))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.cuda(non_blocking = True)
            targets = targets.cuda(non_blocking = True)
            logits, expected_flop = network(inputs)
            loss = criterion(logits, targets)
            
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk = (1,5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(prec1.item(), inputs.size()[0])
            top5.update(prec5.item(), inputs.size()[0])

            if i % print_freq == 0 or (i+1) == len(valid_loader):
                print('**Valid** [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(valid_loader)))
    print('**VALID** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f}'.format(top1=top1, top5=top5, error1 = 100-top1.avg, error5 = 100-top5.avg))
    return losses.avg, top1.avg, top5.avg


def basic_train(xloader, network, criterion, scheduler, optimizer, optim_config, extra_info, print_freq):
    loss, acc1, acc5 = procedure(xloader, network, criterion, scheduler, optimizer, 'train', optim_config, extra_info, print_freq)
    return loss, acc1, acc5

def basic_valid(xloader, network, criterion,optim_config, extra_info, print_freq):
    with torch.no_grad():
        loss, acc1, acc5 = procedure(xloader, network, criterion, None, None, 'valid', None, extra_info, print_freq)
        print('**VALID** Prec@1 {:.2f} Prec@5 {:.2f} Error@1 {:.2f} Error@5 {:.2f}'.format(acc1, acc5, 100-acc1, 100-acc5))
    return loss, acc1, acc5

def procedure(xloader, network, criterion, scheduler, optimizer, mode, config, extra_info, print_freq):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    if mode == 'train':
        network.train()
    elif mode == 'valid':
        network.eval()
    else:
        raise ValueError

    for i, (inputs, targets) in enumerate(xloader):
        if mode == 'train':
            scheduler.update(None, 1.0*i / len(xloader))
        inputs = inputs.cuda(non_blocking = True)    
        targets = targets.cuda(non_blocking = True)
        
        if mode == 'train':
            optimizer.zero_grad()

        features, logits = network(inputs)
        if isinstance(logits, list):
            logits, logits_aux = logits
        else:
            logits, logits_aux = logits, None
        loss = criterion(logits, targets)
        
        if config is not None and hasattr(config, 'auxiliary') and config.auxiliary > 0:
            loss_aux = criterion(logits_aux, targets)
            loss += config.auxiliary * loss_aux

        if mode == 'train':
            loss.backward()
            optimizer.step()

        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1,5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == 'valid' and i % print_freq == 0:
            print('**Valid** [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(xloader)))
    return losses.avg, top1.avg, top5.avg




def finetune_train(xloader, network, criterion, scheduler, optimizer, optim_config, extra_info, print_freq):
    loss, acc1, acc5 = finetune_procedure(xloader, network, criterion, scheduler, optimizer, 'train', optim_config, extra_info, print_freq)
    return loss, acc1, acc5

def finetune_valid(xloader, network, criterion,optim_config, extra_info, print_freq):
    with torch.no_grad():
        loss, acc1, acc5 = finetune_procedure(xloader, network, criterion, None, None, 'valid', None, extra_info, print_freq)
        print('**VALID** Prec@1 {:.2f} Prec@5 {:.2f} Error@1 {:.2f} Error@5 {:.2f}'.format(acc1, acc5, 100-acc1, 100-acc5))
    return loss, acc1, acc5

def finetune_procedure(xloader, network, criterion, scheduler, optimizer, mode, config, extra_info, print_freq):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    if mode == 'train':
        network.train()
    elif mode == 'valid':
        network.eval()
    else:
        raise ValueError

    for i, inputs_tuple in enumerate(xloader):
        if mode == 'train':
            inputs, targets, _, __ = inputs_tuple
            scheduler.update(None, 1.0*i / len(xloader))
        elif mode == 'valid':
            inputs, targets = inputs_tuple
        inputs = inputs.cuda(non_blocking = True)
        targets = targets.cuda(non_blocking = True)
        
        if mode == 'train':
            optimizer.zero_grad()

        features, logits = network(inputs)
        if isinstance(logits, list):
            logits, logits_aux = logits
        else:
            logits, logits_aux = logits, None
        loss = criterion(logits, targets)
        
        if config is not None and hasattr(config, 'auxiliary') and config.auxiliary > 0:
            loss_aux = criterion(logits_aux, targets)
            loss += config.auxiliary * loss_aux

        if mode == 'train':
            loss.backward()
            optimizer.step()

        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1,5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == 'valid' and i % print_freq == 0:
            print('**Valid** [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(xloader)))
    return losses.avg, top1.avg, top5.avg


def scratch_train(search_loader, network, criterion, scheduler, base_optimizer, arch_optimizer, optim_config, extra_info, print_freq):
    
    base_losses, arch_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    arch_cls_losses, arch_flop_losses = AverageMeter(), AverageMeter()
    epoch_str, flop_need, flop_weight, flop_tolerant = extra_info['epoch-str'], extra_info['FLOP-exp'], extra_info['FLOP-weight'], extra_info['FLOP-tolerant']

    # train
    network.train()
    print('[Search] : {:}, FLOP-Require={:.2f}, FLOP-Weight={:.2f}'.format(epoch_str, flop_need, flop_weight))
    network.apply(change_key('search_mode', 'search'))
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):
        scheduler.update(None, 1*step / len(search_loader))
        base_inputs = base_inputs.cuda(non_blocking = True)
        base_targets = base_targets.cuda(non_blocking = True)
        arch_inputs = arch_inputs.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking = True)

        # update base parameters
        base_optimizer.zero_grad()
        logits, expected_flop = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        base_optimizer.step()

        # update arch parameters
        arch_optimizer.zero_grad()
        logits, expected_flop = network(arch_inputs)
        flop_cur = network.get_flop('genotype', None, None)
        #flop_cur = network.module.get_flop('genotype', None, None)
        flop_loss, flop_loss_scale = get_flop_loss(expected_flop, flop_cur, flop_need, flop_tolerant)
        loss = criterion(logits, arch_targets)
        arch_loss = loss + flop_loss * flop_weight
        arch_loss.backward()
        arch_optimizer.step()
        
        # record
        prec1, prec5 = obtain_accuracy(logits.data, arch_targets.data, topk = (1,5))
        top1.update(prec1.item(), arch_inputs.size()[0])
        top5.update(prec5.item(), arch_inputs.size()[0])
        arch_losses.update(arch_loss.item(), arch_inputs.size()[0])
        arch_flop_losses.update(flop_loss_scale, arch_inputs.size()[0])
        arch_cls_losses.update(loss.item(), arch_inputs.size()[0])
        base_losses.update(base_loss.item(), base_inputs.size()[0])
    print('**TRAIN** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f}\
            Base_loss:{baseloss:.3f} Arch-loss:{archloss:.3f}, arch_flop_loss:{archflop:.3f} arch_cls_loss:{clsloss:.3f}'.format(top1=top1, top5=top5, error1 = 100-top1.avg, error5 = 100-top5.avg,\
            baseloss=base_losses.avg, archloss=arch_losses.avg, archflop=arch_flop_losses.avg, clsloss=arch_cls_losses.avg))
    print('Current FLOP at {:} is {:}'.format(epoch_str, flop_cur))
    return arch_losses.avg, top1.avg, top5.avg


def scratch_valid(valid_loader, network, criterion, extra_info, print_freq):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    network.eval()
    network.apply(change_key('search_mode', 'search'))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.cuda(non_blocking = True)
            targets = targets.cuda(non_blocking = True)
            logits, expected_flop = network(inputs)
            loss = criterion(logits, targets)
            
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk = (1,5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(prec1.item(), inputs.size()[0])
            top5.update(prec5.item(), inputs.size()[0])

            if i % print_freq == 0 or (i+1) == len(valid_loader):
                print('**Valid** [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(valid_loader)))
    print('**VALID** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f}'.format(top1=top1, top5=top5, error1 = 100-top1.avg, error5 = 100-top5.avg))
    return losses.avg, top1.avg, top5.avg


def rev_train(search_loader, network, criterion, scheduler, base_optimizer, arch_optimizer,optim_config, extra_info, print_freq):
    
    arch_losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_cls_losses, arch_flop_losses = AverageMeter(), AverageMeter()

    # train
    network.train()
    print('[Search] : {:}'.format(epoch_str))
    network.apply(change_key('search_mode', 'search'))
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(search_loader):
        #scheduler.update(None, 1*step / len(search_loader))
        #base_targets = base_targets.cuda(non_blocking = True)
        arch_inputs = arch_inputs.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking = True)
        """
        # update base parameters
        if pretrain:
            network.eval()
            with torch.no_grad():
                network.apply(change_key('search_mode','basic'))
                logits, expected_flop = network(base_inputs)
            base_loss = criterion(logits, base_targets)
        else:
            base_optimizer.zero_grad()
            logits, expected_flop = network(base_inputs)
            base_loss = criterion(logits, base_targets)
            base_loss.backward()
            base_optimizer.step()
        """
        # update arch parameters
        arch_optimizer.zero_grad()
        logits = network(arch_inputs)
        flop_cur = network.get_flop('genotype', None, None)
        #flop_cur = network.module.get_flop('genotype', None, None)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        arch_optimizer.step()
        
        # record
        prec1, prec5 = obtain_accuracy(logits.data, arch_targets.data, topk = (1,5))
        top1.update(prec1.item(), arch_inputs.size()[0])
        top5.update(prec5.item(), arch_inputs.size()[0])
        arch_losses.update(arch_loss.item(), arch_inputs.size()[0])
        arch_flop_losses.update(flop_loss_scale, arch_inputs.size()[0])
        arch_cls_losses.update(loss.item(), arch_inputs.size()[0])
    
    print('**TRAIN** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f}\
            Arch-loss:{archloss:.3f}, arch_flop_loss:{archflop:.3f} arch_cls_loss:{clsloss:.3f}'.format(top1=top1, top5=top5, error1 = 100-top1.avg, error5 = 100-top5.avg,\
            archloss=arch_losses.avg, archflop=arch_flop_losses.avg, clsloss=arch_cls_losses.avg))
    print('Current FLOP at {:} is {:}'.format(epoch_str, flop_cur))
    return arch_losses.avg, top1.avg, top5.avg



def rev_valid(valid_loader, network, criterion, extra_info, print_freq):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    network.eval()
    network.apply(change_key('search_mode', 'search'))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):

            targets = targets.cuda(non_blocking = True)
            logits, expected_flop = network(inputs)
            loss = criterion(logits, targets)
            
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk = (1,5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(prec1.item(), inputs.size()[0])
            top5.update(prec5.item(), inputs.size()[0])

            if i % print_freq == 0 or (i+1) == len(valid_loader):
                print('**Valid** [{:}][{:03d}/{:03d}]'.format(extra_info, i, len(valid_loader)))
    print('**VALID** Prec@1 {top1.avg:.2f} Prec@5 {top5.avg:.2f} Error@1 {error1:.2f} Error@5 {error5:.2f}'.format(top1=top1, top5=top5, error1 = 100-top1.avg, error5 = 100-top5.avg))
    return losses.avg, top1.avg, top5.avg


