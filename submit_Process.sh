#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-steffejr-ab
#SBATCH --mem-per-cpu=1024M
python ProcessTools.py 1000 1000 10 -0.40 -0.40 -0.40 99
