#!/usr/bin/env bash

set -x

ssh-keygen -q -N "" -t rsa -f ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
echo "    StrictHostKeyChecking no                     " | sudo tee -a /etc/ssh/ssh_config
echo "    PasswordAuthentication no                    " | sudo tee -a /etc/ssh/ssh_config

ls -l ~/.ssh
sudo service ssh restart

ssh -vvv localhost