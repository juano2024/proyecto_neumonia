FROM python:latest

RUN apt-get update -y && \
    apt-get install python3-opencv -y 

WORKDIR /home/src

COPY . ./
RUN pip install -r requirements.txt
