#!/bin/bash
#SBATCH --job-name=xterm
#SBATCH --output=xterm_%j.log
#SBATCH --partition=gui
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=

#SBATCH --account=adamginsburg
#SBATCH --qos=adamginsburg

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3gb
#SBATCH --time=24:00:00
date; hostname; pwd;

unset XDG_RUNTIME_DIR



module purge; module load gui/2 xterm

/apps/gui/2.0.0/bin/start_gui_app xterm

date