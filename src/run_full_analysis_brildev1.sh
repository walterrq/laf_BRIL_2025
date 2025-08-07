#!/bin/bash

echo "[INFO] This script will run the non-linearity full analysis for fill 8880. There are harcoded variables in this script that should be changed in order to adapt to your workspace. Please change them if necessary."
echo "[INFO] Sleeping for 20 seconds in order to force you to open the script! :)"
sleep 20

## Variables to Change

### Path variables
IvdMfw="../VdMFramework" # Path to the vdM Framework. Clone from 'https://gitlab.cern.ch/bril/VdMFramework'
Icentral="/brildata/23/" # Path to detector central
Ibeam_central="/brildata/23/" # Path to beam central
Ivdm_central="/brildata/vdmdata23/" # Path to vdM scans central

Oanalysis="analysis" # Output for the whole analysis
# Ouputs for the different analysis steps
Opoggers="poggers"
Odt_collinearity="collinearity"
Ofpm="fpm"
Ofit="fit"
Ocomparison="comparison"

### Other variables
fill=8880
fill_scan="8880_230606034730_230606035606.hd5"
channels="0 1 2 3 10 11 12 14 15"


## Start of Analysis
set -xe

### First Step: Running poggers examples

python examples/poggers_plt.py --central $Icentral --beam-central $Ibeam_central --fill $fill --output "${Oanalysis}/${Opoggers}_plt" 
python examples/poggers_dt.py --central $Icentral --beam-central $Ibeam_central --fill $fill --output "${Oanalysis}/${Opoggers}_dt"

### Second Step: Running DT collinearity

 python dt_collinearity/dt_collinearity.py "${Oanalysis}/${Opoggers}_plt" "${Oanalysis}/${Opoggers}_dt" $fill --output "${Oanalysis}/${Odt_collinearity}/${fill}" --vdm_central ${Ivdm_central}

### Third Step: Running FPM analysis

cd fpm/
python fpm.py --framework_path "../${IvdMfw}" --hd5_path "${Ivdm_central}/${fill}/${fill_scan}" --calib_path calib.csv --output "../${Oanalysis}/${Ofpm}/${fill_scan}/" --fit_function QG --correction noCorr --iterations 6 --use_cached --no_bcid_distinction --corr_flags ""
cd ..

### Fourth Step: Running the fitting on DT and FPM alphas

python simultaneous_fitting/simultaneous_fitting.py $fill "${Oanalysis}/${Opoggers}_plt" $channels DT "${Oanalysis}/${Odt_collinearity}/${fill}/collinearity.json" --output "${Oanalysis}/${Ofit}/dt/${fill}"
python simultaneous_fitting/simultaneous_fitting.py $fill "${Oanalysis}/${Opoggers}_plt" $channels FPM "${Oanalysis}/${Ofpm}/${fill_scan}/no_bcid_distinction_results_SBIR.pickle" --output "${Oanalysis}/${Ofit}/fpm/${fill}"

### Fifth Step: Comparing PLT channels

python plot.py "${Oanalysis}/${Opoggers}_plt" $fill "${Oanalysis}/${Ofit}/dt/${fill}/best_params.pickle" fpm/calib.csv "${Oanalysis}/${Ofpm}/${fill_scan}/no_bcid_distinction_results_SBIR.pickle" --output "${Oanalysis}/${Ocomparison}/dt/${fill}"
python plot.py "${Oanalysis}/${Opoggers}_plt" $fill "${Oanalysis}/${Ofit}/fpm/${fill}/best_params.pickle" fpm/calib.csv "${Oanalysis}/${Ofpm}/${fill_scan}/no_bcid_distinction_results_SBIR.pickle" --output "${Oanalysis}/${Ocomparison}/fpm/${fill}"
