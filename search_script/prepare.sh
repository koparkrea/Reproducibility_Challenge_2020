#!bin/bash
# bash prepare.sh
# datasets="cifar10 / cifar100 / imagenet-1k"
ratios='0.5'
save_dir=./splits

    python3 ./prepare.py --name cifar10 --root ./data/cifar.python --save ${save_dir}/cifar10-${ratios}.pth 
    python3 ./prepare.py --name cifar100 --root ./data/cifar.python --save ${save_dir}/cifar100-${ratios}.pth 
    python3 ./prepare.py --name imagenet-1k --root ./data/ILSVRC2012 --save ${save_dir}/imagenet-1k-${ratios}.pth 
