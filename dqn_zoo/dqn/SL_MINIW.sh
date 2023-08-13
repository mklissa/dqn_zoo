#!/usr/bin/env bash

jobids=()
port=($(seq 4500 1 5126))
seeds=($(seq 1 1 4))
option_probs=(0.9)
envname="MiniWorld-FourRooms-v0"
algo='dceo'

base_folder="results/${envname}/${algo}"
rand_id=$((1000 + RANDOM % 9000))

if [ "$1" != "test" ]; then
    output_folder=$PWD/slurm_outputs/${envname}/${algo}/$(date +"%m-%d-%H:%M")_${rand_id}//
    mkdir -p $output_folder
    echo 'github place holder' > $output_folder/place_holder.txt
    echo $output_folder
fi

count=0
plot=False
for option_prob in ${option_probs[@]}
do
    for seed in ${seeds[@]}
    do
        if [ -f temprun.sh ] ; then
            rm temprun.sh
        fi
        if (($seed % 2 == 1)); then
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
        --num_iterations 200 --num_train_frames 25_000 --num_eval_frames 5_000 \
        --num_options 5 --lap_dim 20 --option_prob ${option_prob} \
        --results_csv_path ${base_folder}/weight_${option_prob}/$(date +"%m-%d-%H:%M")_${rand_id}_${rand_id}/seed${seed}.csv \
        --plot=${plot} --plot_path plots/${envname}/${algo}/weight_${option_prob}/$(date +"%m-%d-%H:%M")_${rand_id}_${rand_id}/seed${seed}/ "
        echo $k >> temprun.sh
        echo $k

        # JOBID=$(eval "sbatch --parsable temprun.sh")
        JOBID=555
        echo "Submitted job $JOBID"

        folder=${base_folder}/weight_${option_prob}/$(date +"%m-%d-%H:%M")_${rand_id}_${rand_id}
        mkdir -p $folder
        echo $JOBID >> $folder/jobinfo.txt
        echo "Description (if any): $1" >> $folder/jobinfo.txt

        jobids+=("$(echo $JOBID)")
        jobids+=("$(echo $k)")

        count=$((count + 1))
    done
done

if [ "$1" != "test" ]; then

    folder="${base_folder}/$(date +"%m-%d-%H:%M")_${rand_id}"
    filename="${folder}/job_info.txt"
    echo "Info in $filename"
    mkdir -p $folder
    touch $filename
    printf "%s\n" "${jobids[@]}" > $filename
    echo "The current date and time is $(date +"%m/%d/%Y %H:%M")" >> $filename
    echo "The randomly generated ID is ${rand_id}" >> $filename
    echo "Description (if any): $1" >> $filename

    message="${rand_id} $1 ${algo} ${envname}"
    echo $message
    git add .
    git commit -m "$message"

fi
