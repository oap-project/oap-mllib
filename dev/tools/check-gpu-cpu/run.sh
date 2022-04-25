#!/usr/bin/env bash
APP_NAME=check-gpu-cpu

mpirun -n 1 ./${APP_NAME} false
