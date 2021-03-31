#!/usr/bin/env bash

set -x

mkdir ~/.ssh
chmod 0700 ~/.ssh

ssh-keygen -q -N "" -t rsa -f ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa

cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "    StrictHostKeyChecking no                     " | sudo tee -a /etc/ssh/ssh_config

ls -ld ~/.ssh
ls -l ~/.ssh

sudo systemctl restart sshd

ssh -vvv localhost