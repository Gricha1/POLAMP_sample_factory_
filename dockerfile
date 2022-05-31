FROM continuumio/miniconda3

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
RUN pip install -U ray
RUN pip3 install -U ray[rllib]
RUN pip3 install wandb
# COPY requirements.txt /tmp/
# RUN pip3 install -r /tmp/requirements.txt

# WORKDIR /home/brian/robot/lab/rl_planning/PRM-RL/
# RUN echo 42
# RUN pwd
# RUN pip install jsonlib
RUN pip3 install moviepy imageio

RUN apt update && apt install build-essential ffmpeg libsm6 libxext6 git -y

ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

ARG CACHEBUST=1
RUN pip install -U pogema && pip freeze | grep pogema
# COPY /EnvLib/ /tmp/EnvLib/
# COPY /planning/ /tmp/planning/
# COPY /configs/ /tmp/configs/
# COPY rllib_ppo.py /tmp/
# # COPY rllib_ddpg.py /tmp/
# COPY tasksInTrain.txt .
# COPY tasksInValidation.txt .
# # RUN ls tmp/configs/
# CMD  python3 /tmp/rllib_ppo.py
# CMD  python3 /tmp/rllib_ddpg.py
