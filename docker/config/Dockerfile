FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y git

RUN apt-get install -y python

RUN apt-get install -y python3-pip


ADD entrypoint.sh /home/entrypoint.sh

RUN chmod 777 /home/entrypoint.sh

WORKDIR /home

RUN git clone https://github.com/marcoaleixo/ScanSSD.git

WORKDIR /home/ScanSSD

RUN pip install -r requirements.txt

#RUN git checkout version2

ENTRYPOINT [ "/home/entrypoint.sh" ]
