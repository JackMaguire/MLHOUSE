FROM ubuntu:bionic

RUN apt-get upgrade
RUN apt-get update

RUN apt-get -y install git python3 python3-pip 

RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install pandas

# For Visualization
RUN pip3 install pydot && apt-get install graphviz

RUN git clone https://github.com/JackMaguire/MLHOUSE.git

# CMD 

#To run on linux: docker run -v /home:/host/home -t -i jbmaguire/mlhouse_train /bin/bash