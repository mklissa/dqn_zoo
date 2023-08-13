#!/usr/bin/env bash

port=($(seq 4500 1 5126))
seeds=($(seq 1 1 20))
envname="Rubiks2x2x2-v0"
option_probs=(0.1 0.5 0.9)
mkdir -p /scratch/mklissa/outputs/dqn_zoo
count=0
for option_prob in ${option_probs[@]}
do
    for seed in ${seeds[@]}
    do
        if [ -f temprun.sh ] ; then
            rm temprun.sh
        fi
        echo "#!/bin/bash" >> temprun.sh
        echo "#SBATCH --account=rrg-bengioy-ad_gpu" >> temprun.sh
        echo "#SBATCH --output=\"/scratch/mklissa/outputs/dqn_zoo/${envname}_seed${seed}_%j.out\"" >> temprun.sh
        echo "#SBATCH --job-name=${envname}_seed${seed}_%j" >> temprun.sh
        echo "#SBATCH --gres=gpu:1" >> temprun.sh
        echo "#SBATCH --nodes=1" >> temprun.sh
        echo "#SBATCH --ntasks=1" >> temprun.sh
        echo "#SBATCH --cpus-per-task=4" >> temprun.sh
        echo "#SBATCH --mem=50G" >> temprun.sh
        echo "#SBATCH --time=1:55:00" >> temprun.sh
        echo "source $HOME/DCEO/bin/activate" >> temprun.sh
        echo "module load cuda/11.4" >> temprun.sh
        echo "module load cudnn/8.2" >> temprun.sh
        echo "cd $HOME/scratch/dqn_zoo/dqn_zoo/dqn/" >> temprun.sh
        k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\"\
        python run_rubiks.py --batch_size 256 --learn_period 4 \
        --option_prob ${option_prob} --num_options 10 --num_eval_frames 500 --lap_dim 2 --option_learning_steps 200_000 --compress_state \
        --results_csv_path results/${envname}/dco/option_prob${option_prob}/seed${seed}.csv --plot_path plots/${envname}/dco/option_prob${option_prob}/"
        echo $k >> temprun.sh
        echo $k
        eval "sbatch temprun.sh"
        rm temprun.sh
        count=$((count + 1))
    done
done
