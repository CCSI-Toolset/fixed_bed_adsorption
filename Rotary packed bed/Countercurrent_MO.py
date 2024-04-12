import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import (
    MeasurementOptimizer,
    SensitivityData,
    MeasurementData,
    CovarianceStructure,
    ObjectiveLib,
)
import pickle
import time


# number of time points for DCM. This is a steady-state model
Nt = 1

# maximum manual measurement number for each measurement
max_manual_num = 10
# minimal measurement interval
min_interval_num = 10.0
# maximum manual measurement number for all measurements
total_max_manual_num = 10

# from here, we reorganize the measurements order 
# this order matters!!!
measure_keys = ["vg", 
                "Tg", 
                "Ts", 
                "P",
                "y_CO2", 
               ]

num_z_grid = 10

# measurement length of every 
#measure_len = [40,800,800,800,800,800,800]
measure_len = [20,200,200,180,200]
measure_len_loc = [20,
                   90, 10, # Tg ads
                   10, 90 ,# Tg des
                   200,
                   90,  # P ads
                   90, # P des
                   90, 10, # y ads
                   10, 90, # y des
                  # 90, 10, # C co2 ads
                  # 10, 90, # C co2 des
                  # 90, 10, # C n2 ads
                  # 10, 90] # C N2 des
                  ]
num_total_measure = sum(measure_len)

#measure_head_row = {'vg': 0, 'Tg': 40, 'Ts': 840, 'P': 1640, 'y_CO2': 2440, 'C_CO2': 3240, 'C_N2': 4040}

measure_head_row = {'vg': 0, 'Tg': 20, 'Ts': 220, 'P': 420, 
                    'y_CO2': 600}

measure_head_row_loc = {'vg-outlet': 0, 
                       
                        'Tg-ads-inside': 20,
                        'Tg-ads-outlet': 110,
                        'Tg-des-outlet': 120,
                        'Tg-des-inside': 130,
                       
                       'Ts-ads-inside': 220, 
                        'Ts-ads-outlet': 310, 
                        'Ts-des-outlet': 320, 
                        'Ts-des-inside': 330,
                        
                        'P-ads-inside': 420, 
                        #'P-ads-outlet': 510, 
                        #'P-des-outlet': 520, 
                        'P-des-inside': 510,

                       'y_CO2-ads-inside': 600, 
                        'y_CO2-ads-outlet': 690, 
                        'y_CO2-des-outlet': 700, 
                        'y_CO2-des-inside': 710,
                        
                       #'C_CO2-ads-inside': 820, 
                       # 'C_CO2-ads-outlet': 910, 
                       # 'C_CO2-des-outlet': 920, 
                       # 'C_CO2-des-inside': 930,
                        
                       #'C_N2-ads-inside': 1020, 
                       # 'C_N2-ads-outlet': 1110, 
                       # 'C_N2-des-outlet': 1120, 
                       # 'C_N2-des-inside': 1130}
                       }

# index of columns of SCM and DCM in Q
static_ind = list(range(num_total_measure))
# this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
all_ind = static_ind
num_total_measure = len(all_ind)




all_names = [] 

for i, item in enumerate(measure_keys): 
     
    if i==0: # vg 

        for k in range(measure_len[i]):
            
            name = [item]

            if k < num_z_grid:
                name.append("ads")
                name.append(str(9))
                name.append(str(k))

            else:
                k-=num_z_grid 
                name.append("des")
                name.append(str(0))
                name.append(str(k))
            
            all_names.append("_".join(name))
            
    elif i==3: # P 
                
        for k in range(measure_len[i]):
            
            name = [item] 
            
            if k < measure_len[i]//2:
                name.append("ads")
                name.append(str(k//num_z_grid))
                name.append(str(k%num_z_grid))
                
            elif k >= measure_len[i]//2:
                k -= measure_len[i]//2
                k += 10 
                name.append("des")
                name.append(str(k//num_z_grid))
                name.append(str(k%num_z_grid))
                
            all_names.append("_".join(name))
    
    else:

    
        for k in range(measure_len[i]):
            
            name = [item]

            if k < measure_len[i]//2: 
                name.append("ads")
                name.append(str(k//num_z_grid))
                name.append(str(k%num_z_grid))

            elif k>= measure_len[i]//2:

                k -= measure_len[i]//2
                name.append("des")
                name.append(str(k//num_z_grid))
                name.append(str(k%num_z_grid))

            all_names.append("_".join(name))
            



static_cost = [500]*measure_len_loc[0] # vg outlet

static_cost.extend([2000]*measure_len_loc[1]) # Tg ads inside
static_cost.extend([500]*measure_len_loc[2]) # Tg outlet
static_cost.extend([500]*measure_len_loc[3]) # Tg inside
static_cost.extend([2000]*measure_len_loc[4]) # Tg ads inside

static_cost.extend([3000]*measure_len_loc[5]) # Ts 

static_cost.extend([2000]*measure_len_loc[6]) # P-inside 
static_cost.extend([2000]*measure_len_loc[7]) # P-inside 


static_cost.extend([500]*measure_len_loc[8]) # yCO2 inside 
static_cost.extend([100]*measure_len_loc[9]) # yCO2 outlet 
static_cost.extend([100]*measure_len_loc[10]) # yCO2 outlet 
static_cost.extend([500]*measure_len_loc[11]) # yCO2 inside 


#static_cost.extend([0]*(200*2)) # CCO2 inside 

dynamic_cost = [0]*num_total_measure

assert len(static_cost)==len(all_names)


def generate_z_list(start, no_outlet=False):
    # o0: 20, 30, ..., 100,(110),120, ..., (210). 
    # z0: 20, 21,...,29.
    # z1 (0.027): 30,31,...39. 130, ..., 139. 
    # ...
    # z8 (0.973): 100, ,,,, 109. 200,...,209
    # z9 for des (0.99): 210-220
    # z9 (1.0): remove, outlet end (110-119) 
    # z0 for des: remove, outlet end (120-129)
    # 120, 121,...,129. 
    z_list = [] 
    
    if no_outlet: # P 
        for pos in range(9):
            head_row = start + pos*10 
            
            new_z_list = []
            
            # add ads 
            for i in range(10):
                new_z_list.append(head_row+i)
                
            for i in range(10):
                new_z_list.append(head_row+i+90)
                
            z_list.append(new_z_list)
        
    else:    
    
        for pos in range(9):# not 10 since we don't include the outlet end

            head_row = start + pos*10

            new_z_list = []

            # add ads 
            for i in range(10): 
                # add ads
                new_z_list.append(head_row + i)

            if pos > 0:
                # add des
                for i in range(10):
                    new_z_list.append(head_row + i + 100)

            z_list.append(new_z_list)

            
            
            if pos == 8:
                
                new_z_list = [] 
                for i in range(10):
                    new_z_list.append(head_row+10+i)
                    
                z_list.append(new_z_list)
                
                new_z_list = [] 
                for i in range(10):
                    new_z_list.append(head_row+20+i)
                    
                z_list.append(new_z_list)
                
                new_z_list = [] 
                for i in range(10):
                    new_z_list.append(head_row+110+i)

                z_list.append(new_z_list)
            
    return z_list 

#print(generate_z_list(420, no_outlet=True))

def generate_o_list(start, no_outlet=False):
    # o0: 20, 30, ..., 100,(110),120, ..., (210). 
    # z0: 20, 21,...,29. 120, 121,...,129. 
    o_list = [] 
    
    if no_outlet:
        ads_des_idx_diff = 90 
        
        for pos in range(10):# not 10 since we don't include the outlet end

            head_row = start + pos 

            new_o_list = []

            # add ads 
            for i in range(18): 
                # add ads
                new_o_list.append(head_row + i*10)

            o_list.append(new_o_list)
        
        
    else:
        ads_des_idx_diff = 100
    
        for pos in range(10):# not 10 since we don't include the outlet end

            head_row = start + pos 

            new_o_list = []

            # add ads 
            for i in range(10): 
                # add ads
                new_o_list.append(head_row + i*10)

            # add des
            for i in range(10):
                new_o_list.append(head_row + i*10 + ads_des_idx_diff)

            o_list.append(new_o_list)
        
    return o_list 


max_lists_opt = []

max_lists_opt.append(list(range(measure_head_row_loc["Tg-ads-inside"], 
                                measure_head_row_loc["Tg-ads-outlet"])) 
                     + list(range(measure_head_row_loc["Tg-des-inside"], 
                                  measure_head_row_loc["Ts-ads-inside"])))

max_lists_opt.append(list(range(measure_head_row_loc["Ts-ads-inside"], 
                                measure_head_row_loc["Ts-ads-outlet"])) 
                     + list(range(measure_head_row_loc["Ts-des-inside"], 
                                  measure_head_row_loc["P-ads-inside"])))

max_lists_opt.append(list(range(measure_head_row_loc["P-ads-inside"], 
                                measure_head_row_loc["y_CO2-ads-inside"]))) 
                     #+ list(range(measure_head_row_loc["P-des-inside"], 
                     #             measure_head_row_loc["y_CO2-ads-inside"])))

max_lists_opt.append(list(range(measure_head_row_loc["y_CO2-ads-inside"], 
                                measure_head_row_loc["y_CO2-ads-outlet"])) 
                     + list(range(measure_head_row_loc["y_CO2-des-inside"], 
                                  num_total_measure)))



max_z_no, max_o_no = 3, 3 # max 3 locations 

measure_head_row = {'vg': 0, 'Tg': 20, 'Ts': 220, 'P': 420, 
                    'y_CO2': 600}

same_z_list = []
same_o_list = []
# vg has no same z list 
# vg has same o list 
#same_o_list.append(list(range(20)))

for key in measure_keys[1:]:
    
    if key == "P":
        no_outlet_opt=True
    else:
        no_outlet_opt=False
    
    key_z_list = generate_z_list(measure_head_row[key], no_outlet=no_outlet_opt)
    key_o_list = generate_o_list(measure_head_row[key], no_outlet=no_outlet_opt)

    same_z_list.extend(key_z_list)
    same_o_list.extend(key_o_list)
        
print(len(same_z_list), len(same_o_list))


## define MeasurementData object
measure_info = MeasurementData(
    all_names,  # name string
    all_ind,  # jac_index: measurement index in Q
    static_cost,  # static costs
    dynamic_cost,  # dynamic costs
    min_interval_num,  # minimal time interval between two timepoints
    max_manual_num,  # maximum number of timepoints for each measurement
    total_max_manual_num,  # maximum number of timepoints for all measurement
)

# create data object to pre-compute Qs
# read jacobian from the source csv
# Nt is the number of time points for each measurement
jac_info = SensitivityData("./jaco_and_var_10t10/jacobian_10_10", Nt)
static_measurement_index = list(range(num_total_measure))  # the index of CA, CB, CC in the jacobian array, considered as SCM
jac_info.get_jac_list(
    static_measurement_index,  # the index of SCMs in the jacobian array
    None, # No DCMs in this problem
)  # the index of DCMs in the jacobian array


def load_pickle(name): 
    file = open(name, "rb")
    data = pickle.load(file)
    file.close()
    return data

error_cov = load_pickle("./jaco_and_var_10t10/variance_10_10")


# use MeasurementOptimizer to pre-compute the unit FIMs
calculator = MeasurementOptimizer(
    jac_info,  # SensitivityData object
    measure_info,  # MeasurementData object
    error_cov=error_cov,  # error covariance matrix
    error_opt=CovarianceStructure.variance,  # error covariance options
    print_level=1,  # I use highest here to see more information
)

# calculate a list of unit FIMs
calculator.assemble_unit_fims()


initial_solution = {500: "./initial/test_500",
                    1000: "./initial/test_1000",
                    1500: "./initial/test_1500",
    		2000: "./initial/test_2000",
		2500: "./initial/test_2500", 
		3000: "./initial/test_3000", 
		3500: "./initial/test_3500",
		4000: "./initial/test_4000", 
		4500: "./initial/test_4500",
		5000: "./initial/test_5000",
                    6000: "./initial/test_6000",
                    7000: "./initial/test_7000",
                #    10000: "./old_files/test_10000", 
                #    15000: "./old_files/test_15000",
                #    20000: "./old_files/test_20000"}
			}

### MO optimization framework
### MO optimization framework

# optimization options
mip_option = True
objective = ObjectiveLib.A
fixed_nlp_opt = False
mix_obj_option = False
alpha_opt = 0.9

sparse_opt = True
fix_opt = False
small_element = 0.0001  # the small element added to the diagonal of FIM
file_store_name = "updated_"

max_num_z_opt = 3 
max_num_o_opt = 3 
max_num_z_lists_opt = same_z_list
max_num_o_lists_opt = same_o_list

#num_dynamic_time = np.linspace(0, 60, 9)

# map the timepoint index to its real time
#dynamic_time_dict = {}
#for i, tim in enumerate(num_dynamic_time[1:]):
#    dynamic_time_dict[i] = np.round(tim, decimals=2)

# give range of budgets for this case
#budget_ranges = np.linspace(1000, 5000, 11)
# give a trial ranges for a test; we use the first 3 budgets in budget_ranges
#trial_budget_ranges = budget_ranges[:3]irst 3 budgets in budget_ranges
#trial_budget_ranges = budget_ranges[:3]

# ===== run a test for a few budgets =====

# use a starting budget to create the model
start_budget =8000
# timestamp for creating pyomo model
t1 = time.time()
# call the optimizer function to formulate the model and solve for the first time
# optimizer method will 1) create the model and save as self.mod 2) initialize the model
calculator.optimizer(
    start_budget,  # budget
    initial_solution,  # a collection of initializations
    mixed_integer=mip_option,  # if relaxing integer decisions
    obj=objective,  # objective function options, A or D
    mix_obj=mix_obj_option,  # if mixing A- and D-optimality to be the OF
    alpha=alpha_opt,  # the weight of A-optimality if using mixed obj
    fixed_nlp=fixed_nlp_opt,  # if it is a fixed NLP problem
    fix=fix_opt,  # if it is a squared problem
    upper_diagonal_only=sparse_opt,  # if only defining upper triangle part for symmetric matrix
    #multi_component_pairs = multi_component_pairs_opt,
    max_num_z = max_num_z_opt, 
    max_num_o = max_num_o_opt, 
    max_num_z_lists = max_num_z_lists_opt,
    max_num_o_lists = max_num_o_lists_opt,
    max_lists = max_lists_opt,
    #num_dynamic_t_name=num_dynamic_time,  # number of time points of DCMs
    #static_dynamic_pair=static_dynamic,  # if one measurement can be both SCM and DCM
    #time_interval_all_dynamic=time_interval_for_all,  # time interval for time points of DCMs
    fim_diagonal_small_element=small_element,  # a small element added for FIM diagonals to avoid ill-conditioning
    print_level=1,
)  # print level for optimization part
# timestamp for solving pyomo model
t2 = time.time()
calculator.solve(mip_option=mip_option, objective=objective)
# timestamp for finishing
t3 = time.time()
print("model and solver wall clock time:", t3 - t1)
print("solver wall clock time:", t3 - t2)
calculator.extract_store_sol(start_budget, file_store_name)