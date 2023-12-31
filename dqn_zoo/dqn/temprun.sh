#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --output="/home/mklissa/scratch/dqn_zoo/dqn_zoo/dqn/slurm_outputs/MiniWorld-MyWayHomeSparse-v0/dceo/10-19-19:08_11972/seed5_%j.out"
#SBATCH --job-name=MiniWorld-MyWayHomeSparse-v0seed5_%j
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=54:00:00
source /home/mklissa/DCEO/bin/activate
module load cuda/11.4
module load cudnn/8.2
cd /home/mklissa/scratch/dqn_zoo/dqn_zoo/dqn/
xvfb-run -n 4504 -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python run_miniworld.py --environment_name MiniWorld-MyWayHomeSparse-v0 --algo dceo --min_replay_capacity_fraction 0.05 --target_network_update_period 40_000 --num_iterations 200 --num_train_frames 25_000 --num_eval_frames 5_000 --num_options 5 --lap_dim 20 --option_prob 0.9 --plot=True --results_csv_path results/MiniWorld-MyWayHomeSparse-v0/dceo/weight_0.9/num_options5/10-19-19:08_11972/seed5.csv --plot_path plots/MiniWorld-MyWayHomeSparse-v0/dceo/weight_0.9/10-19-19:08_11972/seed5/ --uniform_restarts=False --stop_lap_gradient=False
