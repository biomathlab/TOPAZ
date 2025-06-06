#!/bin/bash
#BSUB -W 200  #2880 #1440
#BSUB -n 1
#BSUB -o out.%J
#BSUB -e err.%J

source ~/.bashrc
module load conda   #delete if using system Python 2
conda activate /usr/local/usrapps/floreslab/TDA_venv3
#python ABC_S01_samples.py #1 day took 6 hours so give 7 next time
#python ABC_S02_crockerplot_save.py #2 day took 11 hours so give 12 next time
#python ABC_S03_crockerplot_distance.py #12 hours took 1 hour so give 2 next time
python ABC_S04_p01_tolerance.py #12 hours took 1 min so give 30 min next time
#python ABC_S05_process_sim.py #unsure
#python ABC_S06_process_crocker.py #unsure
conda deactivate

