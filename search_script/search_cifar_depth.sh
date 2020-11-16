#!bin/bash
# bash ./search_script/search_cifar_depth.sh cifar10 ResNet56 CIFARX criteria 0.1 5 0.57 777

set -e
echo script name: $0
echo $# arguments

dataset=$1
model=$2
optim=$3
batch=256
gumbel_min=$4
gumbel_max=$5
expected_FLOP_ratio=$6
rseed=$7
resume=$8
pretrain=$9


# search from scratch
cifar10_resnet56_nono(){
	echo -e 'nono'
	python3 ./search_main.py --dataset ${dataset} \
	--data_path ./data/cifar.python \
	--model_config ./config_utils/cifar-${model}.config \
	--split_path ./splits/${dataset}-0.5.pth \
	--optim_config ./config_utils/${optim}.config \
	--search_shape criteria \
	--procedure scratch \
	--FLOP_ratio ${expected_FLOP_ratio} \
	--FLOP_weight 2 \
	--FLOP_tolerant 0.05 \
	--gumbel_tau_max ${gumbel_max} \
	--gumbel_tau_min ${gumbel_min} \
	--cutout_length -1 \
	--batch_size ${batch} \
	--rand_seed ${rseed} \
	--workers 4 \
	--eval_frequency 1 \
	--print_freq_eval 200 
}

# search from stopped
cifar10_resnet56_yesno(){
	echo -e 'yesno'
	python3 ./search_main.py --dataset ${dataset} \
	--data_path ./data/cifar.python \
	--model_config ./config_utils/cifar-${model}.config \
	--split_path ./splits/${dataset}-0.5.pth \
	--optim_config ./config_utils/${optim}.config \
	--search_shape criteria \
	--procedure scratch \
	--FLOP_ratio ${expected_FLOP_ratio} \
	--FLOP_weight 2 \
	--FLOP_tolerant 0.05 \
	--gumbel_tau_max ${gumbel_max} \
	--gumbel_tau_min ${gumbel_min} \
	--cutout_length -1 \
	--batch_size ${batch} \
	--rand_seed ${rseed} \
	--workers 4 \
	--eval_frequency 1 \
	--print_freq_eval 200 \
	--resume
}

# search from pretrained weight
cifar10_resnet56_noyes(){
	echo -e 'noyes'
	python3 ./search_main.py --dataset ${dataset} \
	--data_path ./data/cifar.python \
	--model_config ./config_utils/cifar-${model}.config \
	--split_path ./splits/${dataset}-0.5.pth \
	--optim_config ./config_utils/${optim}.config \
	--search_shape criteria \
	--procedure scratch \
	--FLOP_ratio ${expected_FLOP_ratio} \
	--FLOP_weight 2 \
	--FLOP_tolerant 0.05 \
	--gumbel_tau_max ${gumbel_max} \
	--gumbel_tau_min ${gumbel_min} \
	--cutout_length -1 \
	--batch_size ${batch} \
	--rand_seed ${rseed} \
	--workers 4 \
	--eval_frequency 1 \
	--print_freq_eval 200 \
	--pretrain
}

cifar10_resnet56_$resume$pretrain
