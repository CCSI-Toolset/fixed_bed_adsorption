# MBDoE and MO analysis for co-current and counter-current RPB 

Authors: Jialu Wang, Ryan Hughes, Debangsu Bhattacharrya, Alexander Dowling 

## Installation instructions 

The following instructions assume you have anaconda installed. We suggest create an environment with the following commands to run code: 

### Step 1: create a new environment 
- create new environment, called for e.g. `rotary`, with `conda` with `Python` version 3.8

`conda create --name rotary python=3.8`

`conda activate rotary`
   
### Step 2: install `IDAES-PSE`
- this step is necessary for running `homotopy` solve for the model 

`pip install idaes-pse` 

`idaes get-extensions`
   
### Step 3: install `Pyomo` from specified branches
- install from the following branch for a modified version of `Pyomo`:

`pip install git+https://github.com/jialuw96/pyomo/tree/RPB_MBDOE`


### Step 4: install `jupyter notebook`
- this is needed only for the draw_figure.ipynb notebook

  `conda install jupyter notebook`

### Software versions we use for the results 

`Python`: 3.8

`IDAES-PSE`: 2.2.0

`Pyomo`: 6.7.0 dev 0

## Code content 

### Co-current flow model 

- `RPB_model_cocurrent.py`: Co-current flow RPB model provided by Ryan, modified for running `k_aug`

- `Cocurrent_flow_MBDoE.ipynb`: Apply `Pyomo.DoE` to the RPB model with `compute_FIM` and `run_grid_search`. `stochastic_program` is also tried but not debugged

### Counter-current flow model (Results are represented in Jialu's thesis chapter 4)

- `RPB_model_countercurrent_kaug.py`: Conter-current flow RPB model provided by Ryan, modified for running `k_aug`

- `Countercurrent_MBDoE.ipynb`: MBDoE analysis applying `Pyomo.DoE` and `k_aug` to the RPB model with `compute_FIM` and `run_grid_search`. `stochastic_program` is also tried but not debugged

- `Counterflow_MO_data_process.ipynb`: Process the sensitivity data and reformulate the data structure for MO analysis 

- `countercurrent_MO.py`: Measurement optimization analysis to the RPB model

- `counterflow_finite_difference_analysis.ipynb`: MBDoE analysis to the RPB model with finite difference method, not using `Pyomo.DoE`

- `draw_figure.ipynb`: Generates all results figures in the manuscript

## Results content 

### Cocurrent flow model

- `Cocurrent_results/MO_results`: folder, contains the MO results at different budgets. For e.g., Sep17_2000_a means budget=2000, optimized with A-optimality 

- `Cocurrent_results/MBDoE_results`: folder, contains the process models ran with different variables. For e.g., Nov25_368_388 means ads.T = 368K, des.T=388K 

- `Countercurrent_results/MO_MILP_solution`: folder, contains the MO results at different budgets. For e.g., updated_2000 means budget=2000, optimized with A-optimality 

- `Countercurrent_results/MBDoE_results/MBDOE_sens`: folder, contained the process models ran with different variables. For e.g., .Tgin_des_391/Tgin_ads_368 means ads.T = 368K, des.T=391K


## Run Models

Model code and how to run them are provided by Ryan Hughes in `Ryan-Hughes-8/fixed_bed_adsorption` repository: 

Co-current flow model: https://github.com/Ryan-Hughes-8/fixed_bed_adsorption

Counter-current flow model:  https://github.com/Ryan-Hughes-8/fixed_bed_adsorption/tree/counter-current-configuration

## Run measurement optimization results 

The rerun instructions are for counter-current flow model, which is the model we use for the paper / thesis results. 

### Step 1: Achieve sensitivity information (Jacobian) and error variance matrix

- Run `Counterflow_MO_data_process.ipynb` to generate all the sensitivity data

### Step 2: Process the data for MO analysis

- Run `Counterflow_Jacobian_organize.ipynb` to organize the sensitivity data to .csv files that MO code use

### Step 3: Run MO analysis 

- In `Countercurrent_MO.py`, find the toggles in line 17 - 27
- Choose mixed-integer option by `mip_option`
- Choose objective by `objective`
- Choose budget set. It is default to be the budget ranges we use for the paper, from 2000 to 10000

## Run MBDoE analysis

### Run MBDoE with `k_aug` 

- `Countercurrent_MBDoE.ipynb`: MBDoE analysis applying `Pyomo.DoE` and `k_aug` to the RPB model with `compute_FIM` and `run_grid_search`

### Run MBDoE with finite difference 

- `Counterflow-finite-difference-analysis.ipynb`: MBDoE analysis applying finite difference method to the RPB model




