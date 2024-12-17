# import libraries
from RPB_model_kaug import *
import pyomo.environ as pyo

# create pyomo model
RPB = full_model_creation(lean_temp_connection=True, configuration="counter-current")

RPB.ads.P.setub(1.26)
RPB.ads.P_in.setub(1.26)

RPB.des.P.setub(1.04)

RPB.ads.L.fix(7.811853)
RPB.ads.theta.fix(0.606758)
RPB.des.P_in.fix(1.034350)
RPB.ads.Tx.fix(347.700154)
RPB.des.Tx.fix(433)
# RPB.ads.w_rpm.fix(0.003502)
RPB.ads.P_in.fix(1.250714)

RPB.obj = Objective(expr=0)

# initialize using BlockTriangularizationInitializer() with a list of values for initialization factors within the models
init_routine_1(
    RPB, homotopy_points=[1e-5, 1e-4, 1e-3, 1e-2] + np.linspace(0.1, 1, 5).tolist()
)


init_obj = BlockTriangularizationInitializer()

init_obj.config.block_solver_call_options = {"tee": True}
init_obj.config.block_solver_options = {
    # "halt_on_ampl_error": "yes",
    "max_iter": 1000
}

# target = 0.003502
targets = [0.1, 0.05, 0.01, 0.005, 0.003502]

for target in targets:
    print("======target:", target, "=======")
    steps = np.linspace(0, 1, 10)

    points = [(target - RPB.ads.w_rpm()) * i + RPB.ads.w_rpm() for i in steps]

    for i in points:
        print("=====target, i:", target, i, "======")
        RPB.ads.w_rpm.fix(i)

        init_obj.initialization_routine(RPB)

    solver = SolverFactory("ipopt")
    solver.options = {"max_iter": 500, "bound_push": 1e-22, "halt_on_ampl_error": "yes"}
    solver.solve(RPB, tee=True).write()


# save model
to_json(RPB, fname="kaug test full model mar7.json.gz", gz=True, human_read=False)
