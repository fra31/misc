#!/bin/bash

norm='Linf' #'L2'
eps=8  #.5
model_dir='../../../robustbench-addmodels/robustbench/models/' # path to robustbench model zoo

# models: rebuffi_orig -> base model from Rebuffi et al. (2021)
#         rebuffi_sodef -> base model + SODEF

# autoattack on the base model
: 'CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack eval_fast --eot_test 1 --batch_size 150 --eps $eps --model rebuffi_orig --norm $norm --model_dir $model_dir

CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack eval_fast --eot_test 1 --batch_size 150 --eps $eps --model rebuffi_sodef --norm $norm --model_dir $model_dir

# apgd on the base model
loss='ce' # choices: 'cw', 'ce', 'dlr-targeted'
n_restarts=1 # apgd random restarts
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack apgd --eot_test 1 --batch_size 150 --eps $eps --model rebuffi_orig --norm $norm --loss $loss --n_iter 100 --n_restarts $n_restarts

# feature attack on the sodef model
n_iter=100 # apgd iterations
n_restarts=10
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack apgd --eot_test 1 --batch_size 150 --eps $eps --model rebuffi_sodef --norm $norm --loss l1 --n_iter $n_iter --n_restarts $n_restats

# worst-case among AA, transfer apgd from base model, feature attack
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack find_wc --eot_test 1 --batch_size 1000 --eps $eps --model rebuffi_sodef --norm $norm
'

# test found points
echo "AA on base model from Rebuffi et al. (2021)"
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack test_points --eot_test 2 --batch_size 1000 --model rebuffi_orig --model_dir $model_dir --data_path ./results/rebuffi_orig/eval_fast_1_1000_Linf_eps_0.03137_short_second.pth

echo "transfer AA"
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack test_points --eot_test 2 --batch_size 1000 --model rebuffi_sodef --model_dir $model_dir --data_path ./results/rebuffi_orig/eval_fast_1_1000_Linf_eps_0.03137_short_second.pth

echo "worst-case among different attacks on Rebuffi et al. with SODEF defense"
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack test_points --eot_test 2 --batch_size 1000 --model rebuffi_sodef --model_dir $model_dir --data_path ./results/rebuffi_sodef/find_wc_1_1000_Linf_eps_0.03137.pth

