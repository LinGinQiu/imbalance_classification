#!/bin/bash
#SBATCH -A ecsstaff
#SBATCH --partition=ecsall
#SBATCH --job-name=test_script
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # change this if you want more than 1 gpu
#SBATCH --mem=32G
#SBATCH --time=4:00:00

echo Test script 
nvidia-smi

module load conda/py3-latest

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/local/software/conda/miniconda-py3-new/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh" ]; then
        . "/local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh"
    else
        export PATH="/local/software/conda/miniconda-py3-new/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate diff_clip_env

#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr makeup --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 300 --n_inv_step 40 --n_train_step 9 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path celeba_hq.ckpt --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr zombie --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 600 --n_inv_step 40 --n_train_step 9 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0 --l1_loss_w 0 --model_path celeba_hq.ckpt --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt


python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr angry --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5  --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr surprised --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr sad --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr smiling --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr neutral --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr scared --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr disgusted --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr contemptuous --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr pained --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr sleepy --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt
#python main.py --clip_finetune --config celeba.yml --exp ./runs/test --edit_attr happy --do_train 1 --do_test 1 --n_train_img 50 --n_test_img 10 --n_iter 5 --t_0 500 --n_inv_step 40 --n_train_step 6 --n_test_step 40 --lr_clip_finetune 8e-6 --id_loss_w 0.3 --l1_loss_w 0.3 --model_path ckpt_200.pth --clip_model_name /home/hgm1g14/DiffusionCLIP/ViT-B-16.pt


nvidia-smi
echo Test script completed =3

