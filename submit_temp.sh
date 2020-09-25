#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-steffejr-ab
#SBATCH --mem-per-cpu=1024M
echo "Hello World!"
python temp.py
