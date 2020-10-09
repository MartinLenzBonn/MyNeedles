#!/bin/sh
nohup mpirun -np 2 python3 solve.py > log.$(TZ=Europe/Berlin date '+%Y-%m-%d_%H.%M.%S').txt
