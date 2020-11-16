#!bin/bash
# bash ./search_script/basic_limit.sh cifar10 ResNet56 Cifar_basic basic 777 


set -e
echo script name: $0

dataset=$1
model=$2
optim=$3
batch=128
rseed=$4
resume=$5


cifar10_resnet56_no(){
echo -e 'cifar10_resnet56_no'
python3 ./basic_limit.py --dataset ${dataset} \
--data_path ./data/cifar.python \
--model_config ./config_utils/cifar-${model}.config \
--optim_config ./config_utils/${optim}.config \
--search_shape limit \
--procedure basic \
--cutout_length -1 \
--batch_size ${batch} \
--rand_seed ${rseed} \
--workers 4 \
--eval_frequency 1 \
--print_freq_eval 200 
}

cifar10_resnet56_yes(){
echo -e 'cifar10_resnet56_yes'
python3 ./basic_limit.py --dataset ${dataset} \
--data_path ./data/cifar.python \
--model_config ./config_utils/cifar-${model}.config \
--optim_config ./config_utils/${optim}.config \
--search_shape limit \
--procedure basic \
--cutout_length -1 \
--batch_size ${batch} \
--rand_seed ${rseed} \
--workers 4 \
--eval_frequency 1 \
--print_freq_eval 200 \
--resume
}

cifar10_resnet56_$resume
