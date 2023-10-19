#!/usr/bin/env bash


seeds=($(seq 1 1 5))
num_options=(5)
option_probs=(0.9)
envname="MiniWorld-FourRooms-v0"
algo='dceo'
base_folder="results/${envname}/${algo}"

rand_id=$((10000 + RANDOM % 90000))
date=$(date +"%m-%d-%H:%M")
if [ "$1" != "test" ]; then
    output_folder=$PWD/slurm_outputs/${envname}/${algo}/${date}_${rand_id}
    mkdir -p $output_folder
    echo $output_folder
fi

count=0
port=($(seq 4500 1 5126))
for num_option in ${num_options[@]}
do
    for option_prob in ${option_probs[@]}
    do
        for seed in ${seeds[@]}
        do

            if [ -f temprun.sh ] ; then
                rm temprun.sh
            fi

            if (($seed % 1 == 0)); then
                plot=True
            else
                plot=False
            fi

            echo "#!/bin/bash" >> temprun.sh
            echo "#SBATCH --account=rrg-bengioy-ad_gpu" >> temprun.sh
            echo "#SBATCH --output=\"${output_folder}/seed${seed}_%j.out\"" >> temprun.sh
            echo "#SBATCH --job-name=${envname}seed${seed}_%j" >> temprun.sh
            echo "#SBATCH --gres=gpu:1" >> temprun.sh
            echo "#SBATCH --nodes=1" >> temprun.sh
            echo "#SBATCH --ntasks=1" >> temprun.sh
            echo "#SBATCH --cpus-per-task=10" >> temprun.sh
            echo "#SBATCH --mem=50G" >> temprun.sh
            echo "#SBATCH --time=12:00:00" >> temprun.sh
            echo "source $HOME/DCEO/bin/activate" >> temprun.sh
            echo "module load cuda/11.4" >> temprun.sh
            echo "module load cudnn/8.2" >> temprun.sh
            echo "cd $HOME/scratch/dqn_zoo/dqn_zoo/dqn/" >> temprun.sh
            k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\" \
            python run_miniworld.py --environment_name ${envname} --algo ${algo}
            --min_replay_capacity_fraction 0.05 --target_network_update_period 40_000 \
            --num_iterations 200 --num_train_frames 25_000 --num_eval_frames 5_000
            --num_options ${num_option} --lap_dim 20 --option_prob ${option_prob} --plot=${plot} \
            --results_csv_path ${base_folder}/weight_${option_prob}/num_options${num_option}/${date}_${rand_id}/seed${seed}.csv \
            --plot_path plots/${envname}/${algo}/weight_${option_prob}/${date}_${rand_id}/seed${seed}/ \
            --uniform_restarts=True --stop_lap_gradient=False"
            echo $k >> temprun.sh
            echo $k

            if [ "$1" != "test" ]; then
                JOBID=$(eval "sbatch --parsable temprun.sh")
            else
                JOBID=555
            fi
            echo "Submitted job $JOBID"

            folder=${base_folder}/weight_${option_prob}/${date}_${rand_id}
            mkdir -p $folder
            echo $JOBID $k >> $folder/jobinfo.txt

            count=$((count + 1))
        done
    done
done

echo "Description (if any): $1" >> $folder/jobinfo.txt

if [ "$1" != "test" ]; then
    message="${rand_id} $1 ${algo} ${envname}"
    echo $message
    git add ..
    git commit -m "$message"
fi


            # --num_iterations 200 --num_train_frames 25_000 --num_eval_frames 5_000 \ 12h
            # --min_replay_capacity_fraction 0.05 --target_network_update_period 40_000 \ 54h