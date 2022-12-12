docker run -it \
           --gpus all \
           --shm-size=500m \
           -v $(dirname $PWD):/home/polamp_sample apollo_rl_server_img
           #-v $PWD:/home/polamp_sample apollo_rl_server_img
           #-v ../:/home/polamp_sample apollo_rl_server_img
           #bash conda run -n POLAMP_samp_fac_env