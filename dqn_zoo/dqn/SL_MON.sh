#!/usr/bin/env bash

jobids=()
port=($(seq 4500 1 5126))
seeds=($(seq 1 1 5))
envname="MonMiniGrid-SixteenRooms-v0"
algo='rnd'
# option_probs=(0.5 0.9)
option_probs=(0.0 0.1 1.0)
mkdir -p /scratch/mklissa/outputs/dqn_zoo
count=0
plot=False
rand_id=$((10000000 + RANDOM % 90000000))
for option_prob in ${option_probs[@]}
do
    for seed in ${seeds[@]}
    do
        if [ -f temprun.sh ] ; then
            rm temprun.sh
        fi
        if [ $seed -eq 1 ]
        then
            plot=True
        else
            plot=False
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
        echo "#SBATCH --time=2:55:00" >> temprun.sh
        echo "source $HOME/DCEO/bin/activate" >> temprun.sh
        echo "module load cuda/11.4" >> temprun.sh
        echo "module load cudnn/8.2" >> temprun.sh
        echo "cd $HOME/scratch/dqn_zoo/dqn_zoo/dqn/" >> temprun.sh
        k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\"\
        python run_monmini.py --environment_name ${envname} --plot=${plot} --num_eval_frames 500 \
        --algo ${algo} --rnd_w ${option_prob} --num_options 0 --lap_dim 512  \
        --results_csv_path results/${envname}/${algo}/weight${option_prob}/${rand_id}/seed${seed}.csv \
        --plot_path plots/${envname}/${algo}/weight${option_prob}/${rand_id}/ "
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

echo "Info in results/${envname}/${algo}/${rand_id}_jobids.txt"
mkdir -p results/${envname}/${algo}/
touch results/${envname}/${algo}/${rand_id}_jobids.txt
printf "%s\n" "${jobids[@]}" > results/${envname}/${algo}/${rand_id}_jobids.txt

message="${rand_id} ${algo} ${envname}"
git add .
git commit -m "$message"