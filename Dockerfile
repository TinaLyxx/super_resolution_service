FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    python3-pip git


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


COPY ./requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements-dev.txt --no-cache-dir

COPY ./inference.py /opt/ml/code/inference.py
COPY ./pipeline_demofusion_sdxl.py /opt/ml/code/pipeline_demofusion_sdxl.py

ENV SAGEMAKER_PROGRAM inference.py

ENTRYPOINT ["python", "/opt/ml/code/inference.py"]