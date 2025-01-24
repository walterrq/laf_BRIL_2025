# laf_BRIL_2025

Repository of the anomaly finder of the PLT.


## Requirements

The tool is thought to be used at CERN lxplus at the `eos` space. It's highly recommended to use a vitual envionment due to the specific version of the modules needed. To install the tool from the scratch with a virtual environment do:

```
python3.9 -m venv <name_of_the_environment>
cd <name_of_the_environment>
source bin/activate
./bin/python3 -m pip install --upgrade pip
git clone https://github.com/tomasate/laf_BRIL_2025.git
cd laf_BRIL_2025/
pip install -r requirements.txt
```

## Instructions

This tool requires pickle files output from [NonLinearity/poggers](https://gitlab.cern.ch/flpereir/nonlinearity/-/tree/master/poggers?ref_type=heads). 

In order to use this tool, run the `runner_laf.py` like this:

```
python runner_laf.py --path <path/to/the/pickle/files> --fill <number_of_the_fill_in_the_given_path>
```

The output plot will be stored in the folder `src/results` with the fill number.
