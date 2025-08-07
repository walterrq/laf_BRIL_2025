# laf_BRIL_2025

Repository of the anomaly finder of the PLT.

## Instalation

This tool is thought to be used with python 3.10.12 If this is not you python version you can used with a virtual environment from conda. You may want to check if you have it already installed by trying

```
conda --version
```

If you get something similar to `conda: command not found`, you don't have conda installed, and can proceed with the instalation of it. Also if you have conda but it's not installed in the `eos` space, is highly recommended you to reinstall it there, since the needed modules will end filling completely your available space at `afs`. 

### Conda instalation

If you're working at the lxplus, you may want to install conda (and most of your instalations) on the `eos` space since you don't have too much space at the `afs` area. To do this, go to your `.bashrc` and paste the next line at the end of it 

```
export HOME="/eos/user/<user_initial>/<user_name>"
```

Save the changes and do the following command on your terminal:

```
source .bashrc
```

Now the changes were applied. You may want to change to the eos to proceed with the instalation, you can do it by doing on the terminal:

```
cd /eos/user/<user_initial>/<user_name>
```

Now download the conda installer:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

We must give permitions and execute the `.sh` file downloaded

```
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

you may want to accept the terms and if it asks if you want to add conda to your `.bashrc`, you may say yes. Otherways, you'll have to activate conda before every time you want to use it.

Refresh the changes of the `.bashrc` by doing:

```
source /afs/cern.ch/user/<user_initial>/<user_name>/.bashrc
```

Check the instalation once again by doing:

```
conda --version
```

If for any reason you still don't see conda installed, you may want to source to the `.bashrc` at your `eos` space.

### Installing python 3.10.12

Now that you have conda do:

```
conda create -n .laf python=3.10.12
```

Once created, you'll be able to activate it by:

```
conda activate .laf
```

## Installing the requirements

The tool is thought to be used at CERN lxplus at the `eos` space. Clone the repository by doing:

```
git clone https://github.com/walterrq/laf_BRIL_2025.git laf
```

Once you have cloned it cd into it and install the requirements

```
cd laf
pip install -r requirements.txt
```


## Instructions

This tool requires pickle files output from [NonLinearity/poggers](https://gitlab.cern.ch/flpereir/nonlinearity/-/tree/master/poggers?ref_type=heads). 

In order to use this tool, run the `runner_laf.py` like this:

```
python examples/poggers_plt.py --central /eos/cms/store/group/dpg_bril/comm_bril/<year>/physics/ --beam-central /eos/cms/store/group/dpg_bril/comm_bril/2023/physics/ --fill 8873 --output <path_to_output_csv> --dt_path <path_to_the_dt_pickle_files_But_not_necesary> --year <year>
```

You can also use the argument `--out` to set the output folder where the `restults` will be `stored.

The output plot will be stored in the folder `src/results/<year>` with the fill number.
