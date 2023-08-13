#!/usr/bin/env bash

jobids=()
port=($(seq 4500 1 5126))
seeds=($(seq 1 1 3))
option_probs=(0.1 0.5 0.9)
envname="montezuma_revenge"
mkdir -p /scratch/mklissa/outputs/dqn_zoo
count=0
algo='dceo'
rand_id=$((10000000 + RANDOM % 90000000))
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
        echo "#SBATCH --cpus-per-task=10" >> temprun.sh
        echo "#SBATCH --mem=50G" >> temprun.sh
        echo "#SBATCH --time=48:55:00" >> temprun.sh
        echo "source $HOME/DCEO/bin/activate" >> temprun.sh
        echo "module load cuda/11.4" >> temprun.sh
        echo "module load cudnn/8.2" >> temprun.sh
        echo "cd $HOME/scratch/dqn_zoo/dqn_zoo/dqn/" >> temprun.sh
        k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\"\
        python run_atari.py --environment_name ${envname} --num_train_frames 1_000_000 --num_eval_frames 500_000
        --results_csv_path results/${envname}/${algo}/option_prob${option_prob}/${rand_id}/seed${seed}.csv
        --option_prob ${option_prob} --num_options 5 --lap_dim 20"
        echo $k >> temprun.sh
        echo $k

        JOBID=$(eval "sbatch --parsable temprun.sh")
        echo "Submitted job $JOBID"

        jobids+=("$(echo $JOBID)")
        jobids+=("$(echo $k)")

        rm temprun.sh
        count=$((count + 1))
    done
done

echo "Info in dqn/results/${envname}/${algo}/${rand_id}_jobids.txt"
mkdir -p dqn/results/${envname}/${algo}/
touch dqn/results/${envname}/${algo}/${rand_id}_jobids.txt
printf "%s\n" "${jobids[@]}" > dqn/results/${envname}/${algo}/${rand_id}_jobids.txt

message="${rand_id} ${algo} ${envname}"
git add .
git commit -m "$message"