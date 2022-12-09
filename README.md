"# POLAMP_sample_factory" 
# Integrate with Apollo 
## build docker container & run docker container
cd /docker
bash build.sh
bash run.sh
## launch rl server ( inside docker container )
conda activate POLAMP_samp_fac_env
cd home/polamp_sample/
python rl_trajectory_server.py --algo APPO --env=polamp_env --experiment=weights_for_use_1 --continuous_actions_sample False
