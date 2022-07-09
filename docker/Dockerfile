FROM continuumio/miniconda3


RUN apt-get update 
RUN apt install -y cmake libz-dev
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /tmp/
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml
#RUN pip3 install -r /tmp/requirements.txt
#python -m pip install pyyaml


#RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#RUN pip3 install wandb
#COPY /home
# COPY requirements.txt /tmp/
# RUN pip3 install -r /tmp/requirements.txt


#RUN apt update && apt install build-essential ffmpeg libsm6 libxext6 git -y

#ADD requirements.txt /tmp/
#RUN pip install -r /tmp/requirements.txt

#ARG CACHEBUST=1
#RUN pip install -U pogema && pip freeze | grep pogema
