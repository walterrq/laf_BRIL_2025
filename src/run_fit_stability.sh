#!/bin/bash

echo "[INFO] This script will test the stability of the simultaneous fit"
echo "[INFO] Sleeping for 20 seconds in order to force you to open the script! :)"
# sleep 20

## Variables to Change

### Path variables
IvdMfw="../VdMFramework" # Path to the vdM Framework. Clone from 'https://gitlab.cern.ch/bril/VdMFramework'
Icentral="/brildata/23/" # Path to detector central
Ibeam_central="/brildata/23/" # Path to beam central
Ivdm_central="/brildata/vdmdata23/" # Path to vdM scans central

Oanalysis="analysis" # Output for the whole analysis
# Ouputs for the different analysis steps
Opoggers="poggers"
Ostability="stability"

### Other variables
fill=8880
fill_scan="8880_230606034730_230606035606.hd5"
channels="0 1 2 3 11 12 15"


## Start of Analysis
set -xe

### First Step: Running poggers examples

python examples/poggers_plt.py --central $Icentral --beam-central $Ibeam_central --fill $fill --output "${Oanalysis}/${Opoggers}_plt"

### Second Step: Running fit stability

python simultaneous_fitting/isimfit.py $fill "${Oanalysis}/${Opoggers}_plt" $channels --n-trials 200 --starts 100 --use-log-form --output "${Oanalysis}/${Ostability}d" 