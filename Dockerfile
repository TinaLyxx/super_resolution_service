FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*



COPY ./requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --no-cache-dir

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml:${PATH}"

COPY codes /opt/ml
WORKDIR /opt/ml
