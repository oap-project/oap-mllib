#!/usr/bin/env bash

sudo apt-get update -q
sudo apt-get install -y -q python3-pip python3-setuptools python3-wheel

pip3 install --user numpy==1.21.2 -q

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 10

echo python is in $(which python)
python --version

echo python3 is in $(which python3)
python3 --version