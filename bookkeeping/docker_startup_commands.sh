#!/bin/bash

apt-get upgrade
apt-get update
apt-get install git
apt-get install python3
apt-get -y install python3-pip
git clone https://github.com/JackMaguire/MLHOUSE.git

pip3 install tensorflow
pip3 install keras
pip3 install pandas

#For visualization
# pip3 install pydot
# apt-get install graphviz
