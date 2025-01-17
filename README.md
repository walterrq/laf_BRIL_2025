# laf_BRIL_2025

Repository of the anomaly finder of the PLT.

## Instructions

If it's the first time you use the laf tool make sure you install the requirement libraries. Once you clone the repo do

```
pip install -r requirements.txt
```

This tool requires pickle files output from [NonLinearity/poggers](https://gitlab.cern.ch/flpereir/nonlinearity/-/tree/master/poggers?ref_type=heads). 

In order to use this tool, run the `runner_laf.py` like this:

```
python runner_laf.py --pickles_path <path/to/the/pickle/files> --fill_number <number_of_the_fill_in_the_given_path>
```

The output plot will be stored in the folder `results` with the fill number.
