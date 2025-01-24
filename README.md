# laf_BRIL_2025

Repository of the anomaly finder of the PLT.

## Instructions

If it's the first time you use the laf tool make sure you install the requirement libraries. The process to install the is the same that uses [PLT Offline Repo](https://github.com/cmsplt/PLTOffline), and the instructions can be found in the [laf document](https://docs.google.com/document/d/1jXkUQ4Mt5PFmiV5IuQkr71Nw3jQUUcfnVggKCsIzocM/edit?tab=t.0)

This tool requires pickle files output from [NonLinearity/poggers](https://gitlab.cern.ch/flpereir/nonlinearity/-/tree/master/poggers?ref_type=heads). 

In order to use this tool, run the `runner_laf.py` like this:

```
python runner_laf.py --pickles <path/to/the/pickle/files> --fill <number_of_the_fill_in_the_given_path>
```

The output plot will be stored in the folder `results` with the fill number.
