#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=test_script
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # change this if you want more than 1 gpu
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-user=cq2u24@soton.ac.uk

echo Test script 
nvidia-smi

module load conda/python3

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

conda activate imb

python main.py --run


