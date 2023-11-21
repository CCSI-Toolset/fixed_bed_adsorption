import numpy as np
import pandas as pd
import pyomo.environ as pyo
from measure_optimize import MeasurementOptimizer, DataProcess, CovarianceStructure, ObjectiveLib
#import matplotlib.pyplot as plt
import pickle 

Nt = 1

# they are set up but not work
max_manual_num = 100
min_interval_num = 100

#measure_names = ["y_CO2", "C_CO2", "C_N2", "Tg", "Pg", "vg", "q_CO2","Ts", "y_theta"]
measure_names = ["Tg", "Pg", "vg", "Ts", "y_CO2",  "C_CO2", "C_N2",  "q_CO2","y_theta"]
num_measure = len(measure_names)

z_pos = 8
z_names = "01234567"

o_pos = 3 
o_names = "012"

num_mod = 2 
mod_names = "ads", "des"

num_total = num_measure*z_pos*o_pos*num_mod

all_ind = list(range(num_total))


# name generator
# CCO2, ADS, Z=0, O=0
# CCO2, ADS, Z=1, O=0
# ...
# CCO2, ADS, Z=0, O=1
#...
# CCO2, DES, Z=0, O=0
# CCO2, DES, Z=1, O=0
# ...
# CCO2, DES, Z=0, O=1
# adsorption + desorption, ...

def name_generator():
    
    name_list = []
    static_costs = []
    dynamic_costs = []
    
    
    for m_n in measure_names:
        for mod_n in mod_names:
            if m_n in ["Tg", "Pg", "vg", "Ts"]:
                static = 1000
                dynamic = 0 
            else:
                static = 0 
                dynamic = 100 
            
            for o in list(o_names):
                for z in list(z_names):
                    name = "_".join([m_n, z, o, mod_n])
                    
                    name_list.append(name)
                    static_costs.append(static)
                    dynamic_costs.append(dynamic)
    
    return name_list, static_costs, dynamic_costs

all_names, static_cost, dynamic_cost = name_generator() 
#print(dynamic_cost)

num_z_o = z_pos*o_pos     

max_manual = [max_manual_num]*num_total
min_time_interval = [min_interval_num]*num_total

# identity matrix as cov 
error_cov = [[0]*num_total for _ in range(num_total)]

for _ in range(num_total):
    error_cov[_][_] = 1
    
measure_info = pd.DataFrame({
    "name": all_names,
    "Q_index": all_ind,
        "static_cost": static_cost,
    "dynamic_cost": dynamic_cost,
    "min_time_interval": min_time_interval, 
    "max_manual_number": max_manual
})

print(measure_info)


dataObject = DataProcess()
dataObject.read_jacobian('./MO_FIMs/Q_z8')
Q = dataObject.get_Q_list(None, all_ind, Nt)

print(np.shape(Q))

print(Q[1])


calculator = MeasurementOptimizer(Q, measure_info, error_cov=error_cov, error_opt=CovarianceStructure.measure_correlation, verbose=True)


fim_expect = calculator.fim_computation()

print(np.shape(calculator.fim_collection))


num_static = 4*num_mod*z_pos*o_pos
num_dynamic  = 5*num_mod*z_pos*o_pos

print(num_static, num_dynamic)
num_total = num_static+num_dynamic


#with open('./kinetics_results/May2_3000_a', 'rb') as f:
#    init_cov_y = pickle.load(f)
#print(init_cov_y)

fim_prior = [[0]*5 for _ in range(5)]

for _ in range(5):
    fim_prior[_][_] = 1 
    
fim_prior = np.asarray(fim_prior)

mip_option = False
objective = ObjectiveLib.A
sparse_opt = True
fix_opt = False

manual_num = 100
budget_opt = 5000

total_manual_init = 0
dynamic_install_init = [1]*num_total 

num_dynamic_time = [0]

#static_dynamic = [[0,3],[1,4],[2,5]]
#time_interval_for_all = True

#dynamic_time_dict = {}
#for i, tim in enumerate(num_dynamic_time[1:]):
#    dynamic_time_dict[i] = np.round(tim, decimals=2)
    
#print(dynamic_time_dict)


mod = calculator.continuous_optimization(mixed_integer=mip_option, 
                      obj=objective, 
                    fix=fix_opt, 
                    upper_diagonal_only=sparse_opt, 
                    num_dynamic_t_name = num_dynamic_time, 
                    manual_number = manual_num, 
                    budget=budget_opt,
                    #init_cov_y= init_cov_y,
                    initial_fim = fim_prior,
                    #dynamic_install_initial = dynamic_install_init, 
                    #static_dynamic_pair=static_dynamic,
                    #time_interval_all_dynamic = time_interval_for_all,
                    total_manual_num_init=total_manual_init)

mod = calculator.solve(mod, mip_option=mip_option, objective = objective)

fim_result = np.zeros((5,5))
for i in range(5):
    for j in range(i,5):
        fim_result[i,j] = fim_result[j,i] = pyo.value(mod.TotalFIM[i,j])
        
print(fim_result)  
print('trace:', np.trace(fim_result))
print('det:', np.linalg.det(fim_result))
print(np.linalg.eigvals(fim_result))

print("Pyomo OF:", pyo.value(mod.Obj))
print("Log_det:", np.log(np.linalg.det(fim_result)))

ans_y, sol_y = calculator.extract_solutions(mod)
print('pyomo calculated cost:', pyo.value(mod.cost))
print("if install dynamic measurements:")
print(pyo.value(mod.if_install_dynamic[3]))
print(pyo.value(mod.if_install_dynamic[4]))
print(pyo.value(mod.if_install_dynamic[5]))
