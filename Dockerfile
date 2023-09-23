FROM ubuntu:22.04

WORKDIR /solution
COPY . .
COPY weights/best.pt ./weights
# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 && pip3 install -U pip && \
    pip3 install --upgrade pip && pip3 install -r requirements.txt && mkdir -p ./weights && \
    mkdir -p ./private/images && mkdir -p ./private/labels && mkdir -p ./output

CMD /bin/sh -c "python3 solution.py && python3 scorer.py"
