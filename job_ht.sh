#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=co_stat
#
# Partition:
#SBATCH --partition=savio2
#
# Tasks per node
#SBATCH --ntasks-per-node=24
#
# Nodes
#SBATCH --nodes=2

# Wall clock limit:
#SBATCH --time=00:30:00
#
## Command(s) to run:
ht_helper.sh -t taskfile -n1 -s1 -v
