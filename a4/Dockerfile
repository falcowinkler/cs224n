FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
ADD . /app
WORKDIR /app
RUN pip install -r gpu_requirements.txt
RUN sh run.sh vocab
ENTRYPOINT sh run.sh train