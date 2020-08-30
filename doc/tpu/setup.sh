#!/bin/bash
#pip3 freeze >> requirements.txt

sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev

wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
tar xvf Python-3.6.9.tgz
cd Python-3.6.9
./configure --enable-optimizations --enable-shared --with-ensurepip=install
make -j8
sudo make altinstall

sudo cp libpython3.6m.so.1.0 /usr/lib/

sudo apt install libcurl4-openssl-dev libssl-dev
sudo pip3.6 install --upgrade pip
pip3.6 install -r requirements_tpu.txt