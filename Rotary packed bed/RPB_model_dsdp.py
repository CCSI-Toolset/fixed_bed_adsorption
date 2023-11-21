from RPB_model_for_debug import *
import pyomo.environ as pyo
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp


# Create model instance
m=RPB_model(mode="desorption", kaug=True)

m.hgx.fix()
m.Obj = pyo.Objective(expr=0, sense=pyo.minimize)

m.P_in.fix(1.1)
m.Tg_in.fix()
m.y_in.fix()
m.P_out.fix(1.01325)

# solve with ipopt
solver = pyo.SolverFactory("ipopt")
solver.solve(m, tee=True)


### run get_dsdp
param = {"hgx": 25*1e-3}
param_list = ["hgx"]

dsdp, col = get_dsdp(m, param_list, param, tee=True)