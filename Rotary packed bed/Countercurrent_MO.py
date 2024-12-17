### Measurement optimization for Counter-current flow RPB model 

# import libaries
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

# optimization options
# mixed-integer or relaxed
mip_option = True
# Objective, ObjectiveLib.A or .D
objective_opt = ObjectiveLib.A
# the small element added to the diagonal of FIM
small_element = 0.0001 
# file store name 
file_store_name = "updated_"
# all budgets to run MO
budget_set = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000] # paper results 

# problem run function 
def countercurrent_MO(budget_list, mip_option_opt, objective_opt, small_element_opt=0.0001, file_store_name_opt="MO_"):
    """
    Run MO analysis 
    
    Arguments
    ---------
    budget_list: a list of budgets
    mip_option_opt: mixed-integer or relaxed, if True, mixed-integer, if False, relaxed
    objective_opt: choose from ObjectiveLib.A or .D
    small_element_opt: the small element added to the diagonal of FIM, default to be 0.0001
    file_store_name_opt: string of file name
    """
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
    # number of z locations
    num_z_grid = 10

    # measurement length of every measurement
    measure_len = [20,200,200,180,200]
    # length of each measurement locations
    measure_len_loc = [20,
                       90, 10, # Tg ads
                       10, 90 ,# Tg des
                       200,
                       90,  # P ads
                       90, # P des
                       90, 10, # y ads
                       10, 90, # y des
                      ]
    num_total_measure = sum(measure_len)
    # measurement head row index number
    measure_head_row = {'vg': 0, 'Tg': 20, 'Ts': 220, 'P': 420, 
                        'y_CO2': 600}
    # split the head row to ads. and des. sections, inside-the-bed and outlet end 
    measure_head_row_loc = {'vg-outlet': 0, # outlet end measurements start here

                            'Tg-ads-inside': 20, # ads. inside-the-bed measurements start here
                            'Tg-ads-outlet': 110, # ads. outlet end measurements start here
                            'Tg-des-outlet': 120, # des. outlet end measurements start here
                            'Tg-des-inside': 130, # des. inside-the-bed measurements start here

                           'Ts-ads-inside': 220, # ads. inside-the-bed measurements start here
                            'Ts-ads-outlet': 310, # ads. outlet end measurements start here
                            'Ts-des-outlet': 320, # des. outlet end measurements start here
                            'Ts-des-inside': 330, # des. inside-the-bed measurements start here

                            'P-ads-inside': 420, # ads. inside-the-bed measurements start here
                            'P-des-inside': 510,# des. inside-the-bed measurements start here

                           'y_CO2-ads-inside': 600, # ads. inside-the-bed measurements start here
                            'y_CO2-ads-outlet': 690, # ads. outlet end measurements start here
                            'y_CO2-des-outlet': 700, # des. outlet end measurements start here
                            'y_CO2-des-inside': 710, # des. inside-the-bed measurements start here
                           }
    
    # index of columns of SCM and DCM in Q
    static_ind = list(range(num_total_measure))
    # this index is the number of SCM + nubmer of DCM, not number of DCM timepoints
    all_ind = static_ind
    num_total_measure = len(all_ind)

    # initialize the name list
    all_names = [] 
    # loop over measurements 
    for i, item in enumerate(measure_keys): 
        # if the measurement is velocity
        if i==0: # vg 
            # loop over its index ranges
            for k in range(measure_len[i]):
                # initialize its name string
                name = [item]
                # decide if this location is ads. or des.
                if k < num_z_grid:
                    name.append("ads")
                    # 9 is the outlet end location index
                    name.append(str(9))
                    name.append(str(k))
                # decide if this location is ads. or des. 
                else:
                    k-=num_z_grid 
                    name.append("des")
                    # 0 is the outlet end location index
                    name.append(str(0))
                    name.append(str(k))
                # generate string names
                all_names.append("_".join(name))
        # if the measurement is pressure
        elif i==3: # P 
            # loop over its index ranges
            for k in range(measure_len[i]):
                # initialize its name string
                name = [item] 
                # decide if this location is ads. or des.
                if k < measure_len[i]//2:
                    name.append("ads")
                    # get z locations
                    name.append(str(k//num_z_grid))
                    # get theta locations
                    name.append(str(k%num_z_grid))
                # decide if this location is ads. or des.
                elif k >= measure_len[i]//2:
                    k -= measure_len[i]//2
                    k += 10 
                    name.append("des")
                    # get z locations
                    name.append(str(k//num_z_grid))
                    # get theta locations
                    name.append(str(k%num_z_grid))
                # generate string names
                all_names.append("_".join(name))
        # for all other measurements
        else:
            # loop over its index ranges
            for k in range(measure_len[i]):
                # initialize its name string
                name = [item]
                # decide if this location is ads. or des.
                if k < measure_len[i]//2: 
                    name.append("ads")
                    # get z locations
                    name.append(str(k//num_z_grid))
                    # get theta locations
                    name.append(str(k%num_z_grid))
                # decide if this location is ads. or des.
                elif k>= measure_len[i]//2:
                    k -= measure_len[i]//2
                    name.append("des")
                    # get z locations
                    name.append(str(k//num_z_grid))
                    # get theta locations
                    name.append(str(k%num_z_grid))
                # generate string names
                all_names.append("_".join(name))

    ## Specify measurement costs
    # static costs are the same for all locations for one measurement
    # vg outlet
    static_cost = [500]*measure_len_loc[0] 
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

    # dynamic costs are all 0 since they are all specified to be SCMs
    dynamic_cost = [0]*num_total_measure

    # check if # measurements have enough costs specified
    assert len(static_cost)==len(all_names), "Number of measurements are not consistent with number of costs"
    
    # number of inside-the-bed measurement locations 
    num_inside_bed = 9 
    # number of total measurement locations
    num_total_loc = 10 

    def generate_z_list(start, no_outlet=False):
        """
        Generate lists to include all the theta locations for one z location 
        For e.g., 
        all z locations for theta0: 20, 30, ..., 100,(110),120, ..., (210). 
        all theta locations for z0: 20, 21,...,29.
         for z1 (0.027): 30,31,...39. 130, ..., 139. 
         ...
         for z8 (0.973): 100, ,,,, 109. 200,...,209
         for z9 for des (0.99): 210-220
         for z9 (1.0): remove, outlet end (110-119) 
         for z0 for des: remove, outlet end (120-129)
         for 120, 121,...,129.
         
        Arguments
        ---------
        start: 
        no_outlet: if return all generated lists
        """
        # all lists include all the theta locations for one z location 
        z_list = [] 
        # if there is no outlet locations, such as for pressure
        if no_outlet: # P 
            # for each location inside-the-bed. not 10 since we don't include the outlet end
            for pos in range(num_inside_bed):
                # locate head row number 
                head_row = start + pos*num_total_loc 
                # new z list 
                new_z_list = []
                # add ads. section locations
                for i in range(num_total_loc):
                    new_z_list.append(head_row+i)
                # add des. section locations 
                for i in range(num_total_loc):
                    new_z_list.append(head_row+i+num_inside_bed*num_total_loc)

                z_list.append(new_z_list)
        # if there are outlet locations, such as for pressure
        else:    
            # for each location inside-the-bed. not 10 since we don't include the outlet end
            for pos in range(num_inside_bed):# not 10 since we don't include the outlet end
                # locate head row number 
                head_row = start + pos*num_total_loc
                # new z list 
                new_z_list = []
                # add ads. section locations
                for i in range(num_total_loc): 
                    new_z_list.append(head_row + i)
                # add des. section locations 
                if pos > 0:
                    for i in range(num_total_loc):
                        new_z_list.append(head_row + i + num_total_loc**2)
                # append z list
                z_list.append(new_z_list)
                # if location list is 8, meaning the z in ads. and des. sections has different locations, therefore
                # delt with separately 
                if pos == 8:
                    # new z list
                    new_z_list = []
                    # add ads. section locations 
                    for i in range(num_total_loc):
                        new_z_list.append(head_row+num_total_loc+i)
                    # append z list 
                    z_list.append(new_z_list)
                    # new z list
                    new_z_list = [] 
                    for i in range(num_total_loc):
                        new_z_list.append(head_row+num_total_loc*2+i)
                    # append z list 
                    z_list.append(new_z_list)
                    # new z list
                    new_z_list = []
                    # append z locations
                    for i in range(num_total_loc):
                        new_z_list.append(head_row+num_total_loc**2+num_total_loc+i)
                    # append z list 
                    z_list.append(new_z_list)

        return z_list 

    def generate_o_list(start, no_outlet=False):
        """
        Generate lists to include all the theta locations for one z location 
        For e.g., 
        all z locations for theta0: 20, 30, ..., 100,(110),120, ..., (210). 
         
        Arguments
        ---------
        start: 
        no_outlet: if return all generated lists
        """ 
        # initialize lists 
        o_list = [] 
        # if there is no outlet locations, such as for pressure
        if no_outlet:
            # head index difference for ads. and des. sections
            ads_des_idx_diff = num_total_loc*num_inside_bed
            # for each locations
            for pos in range(num_total_loc):
                # locate head row
                head_row = start + pos 
                # initialize list
                new_o_list = []
                # add ads 
                for i in range(num_inside_bed*2): 
                    # add ads locations
                    new_o_list.append(head_row + i*num_total_loc)
                # append new list 
                o_list.append(new_o_list)
        # if there are outlet locations, such as for pressure
        else:
            # head index difference for ads. and des. sections
            ads_des_idx_diff = num_total_loc**2
            # for each locations
            for pos in range(num_total_loc):# not 10 since we don't include the outlet end
                # locate head row
                head_row = start + pos 
                # initialize list
                new_o_list = []
                # add ads 
                for i in range(num_total_loc): 
                    # add ads locations
                    new_o_list.append(head_row + i*num_total_loc)
                # add des
                for i in range(num_total_loc):
                    new_o_list.append(head_row + i*num_total_loc + ads_des_idx_diff)
                # append new list
                o_list.append(new_o_list)

        return o_list 

    # maximum total locations for one measurement
    # initialize the constraint lists
    max_lists_opt = []
    # all locations for Tg
    max_lists_opt.append(list(range(measure_head_row_loc["Tg-ads-inside"], 
                                    measure_head_row_loc["Tg-ads-outlet"])) 
                         + list(range(measure_head_row_loc["Tg-des-inside"], 
                                      measure_head_row_loc["Ts-ads-inside"])))
    # all locations for Ts
    max_lists_opt.append(list(range(measure_head_row_loc["Ts-ads-inside"], 
                                    measure_head_row_loc["Ts-ads-outlet"])) 
                         + list(range(measure_head_row_loc["Ts-des-inside"], 
                                      measure_head_row_loc["P-ads-inside"])))
    # all locations for P
    max_lists_opt.append(list(range(measure_head_row_loc["P-ads-inside"], 
                                    measure_head_row_loc["y_CO2-ads-inside"]))) 
    # all locations for yCO2
    max_lists_opt.append(list(range(measure_head_row_loc["y_CO2-ads-inside"], 
                                    measure_head_row_loc["y_CO2-ads-outlet"])) 
                         + list(range(measure_head_row_loc["y_CO2-des-inside"], 
                                      num_total_measure)))

    # max 3 locations for one theta or one z location
    max_z_no, max_o_no = 3, 3 
    # head row index for each measurement 
    measure_head_row = {'vg': 0, 'Tg': 20, 'Ts': 220, 'P': 420, 
                        'y_CO2': 600}
    # initialize the z list for the maximum locations for one location rule 
    same_z_list = []
    # initialize the theta list for the maximum locations for one location rule 
    same_o_list = []
    # vg has no same z list 
    # vg has same o list 
    #same_o_list.append(list(range(20)))
    # for all measurements excluding vg
    for key in measure_keys[1:]:
        # if pressure, no outlet 
        if key == "P":
            no_outlet_opt=True
        # if not pressure, have outlet locations
        else:
            no_outlet_opt=False
        # z list for the maximum locations for one location rule 
        key_z_list = generate_z_list(measure_head_row[key], no_outlet=no_outlet_opt)
        # o list for the maximum locations for one location rule 
        key_o_list = generate_o_list(measure_head_row[key], no_outlet=no_outlet_opt)
        # z list for the maximum locations for one location rule 
        same_z_list.extend(key_z_list)
        # o list for the maximum locations for one location rule 
        same_o_list.extend(key_o_list)


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
    jac_info = SensitivityData("./Countercurrent_MO_QVs/jacobian_10_10", Nt)
    # the index of CA, CB, CC in the jacobian array, considered as SCM
    static_measurement_index = list(range(num_total_measure))  
    jac_info.get_jac_list(
        static_measurement_index,  # the index of SCMs in the jacobian array
        None, # No DCMs in this problem
    )  # the index of DCMs in the jacobian array


    def load_pickle(name): 
        """read variance matrix from pickle file stored
        """
        file = open(name, "rb")
        data = pickle.load(file)
        file.close()
        return data
    # error covariance matrix
    error_cov = load_pickle("./Countercurrent_MO_QVs/variance_10_10")

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
    # initial solutions
    initial_solution = {2000: "./Countercurrent_results/MO_MILP_solution/updated_2000",
                        2500: "./Countercurrent_results/MO_MILP_solution/updated_2500", 
                        3000: "./Countercurrent_results/MO_MILP_solution/updated_3000", 
                        3500: "./Countercurrent_results/MO_MILP_solution/updated_3500",
                        4000: "./Countercurrent_results/MO_MILP_solution/updated_4000", 
                        4500: "./Countercurrent_results/MO_MILP_solution/updated_4500",
                        5000: "./Countercurrent_results/MO_MILP_solution/updated_5000",
                        6000: "./Countercurrent_results/MO_MILP_solution/updated_6000",
                        7000: "./Countercurrent_results/MO_MILP_solution/updated_7000", 
                        8000: "./Countercurrent_results/MO_MILP_solution/updated_8000",
                        9000: "./Countercurrent_results/MO_MILP_solution/updated_9000",
                        10000: "./Countercurrent_results/MO_MILP_solution/updated_10000", 
                       }

    fixed_nlp_opt = False
    mix_obj_option = False
    alpha_opt = 0.9

    sparse_opt = True
    fix_opt = False

    max_num_z_opt = 3 
    max_num_o_opt = 3 
    max_num_z_lists_opt = same_z_list
    max_num_o_lists_opt = same_o_list

    # run a few budgets 
    for budget in budget_list:
        # timestamp for creating pyomo model
        t1 = time.time()
        # call the optimizer function to formulate the model and solve for the first time
        # optimizer method will 1) create the model and save as self.mod 2) initialize the model
        calculator.optimizer(
            budget,  # budget
            initial_solution,  # a collection of initializations
            mixed_integer=mip_option_opt,  # if relaxing integer decisions
            obj=objective_opt,  # objective function options, A or D
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
            fim_diagonal_small_element=small_element_opt,  # a small element added for FIM diagonals to avoid ill-conditioning
            print_level=1,
        )  # print level for optimization part
        # timestamp for solving pyomo model
        t2 = time.time()
        calculator.solve(mip_option=mip_option, objective=objective)
        # timestamp for finishing
        t3 = time.time()
        print("model and solver wall clock time:", t3 - t1)
        print("solver wall clock time:", t3 - t2)
        calculator.extract_store_sol(budget, file_store_name_opt)
        
countercurrent_MO(budget_set, mip_option, objective_opt)