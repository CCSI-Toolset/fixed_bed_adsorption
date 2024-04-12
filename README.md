# MBDoE and MO analysis for co-current and counter-current RPB 

Authors: Jialu Wang, Ryan Hughes, Debangsu Bhattacharrya, Alexander Dowling 

## Installation instructions 

The following instructions assume you have anaconda installed. We suggest create an environment with the following commands to run code: 

### Step 1: create a new environment 
- create new environment, called for e.g. `rotary`, with `conda` with `Python` version 3.8

`conda create --name measurement_optimization python=3.8`

`conda activate rotary`
   
### Step 2: install `IDAES-PSE`
- this step is necessary for running `homotopy` solve for the model 

`pip install idaes-pse` 

`idaes get-extensions`
   
### Step 3: install `Pyomo` from specified branches
- install from the following branch for a modified version of `Pyomo`:

`pip install git+https://github.com/jialuw96/pyomo/tree/rotary`


### Step 4: install `jupyter notebook`
- this is needed only for the draw_figure.ipynb notebook

  `conda install jupyter notebook`

### Software versions we use for the results 

`Python`: 3.8

`IDAES-PSE`: 2.2.0

`Pyomo`: 6.7.0 dev 0

## Code 

- `measure_optimize.py`: Measurement optimization optimization framework

- `greybox_generalize.py`: Grey-box generalization 

- `kinetics_MO.py`: Kinetics case study

- `rotary_bed_MO.py`: Rotary bed case study

- `draw_figure.ipynb`: Generates all results figures in the manuscript
