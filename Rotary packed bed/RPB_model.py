"""
Created on Thu Feb 16 08:22:36 2023

@author: ryank
"""
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Constraint,
    Expression,
    units,
    exp,
    log,
    TransformationFactory,
    SolverFactory,
    value,
    PositiveReals,
    NonNegativeReals,
    Set,
    Objective,
    Block,
)
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from idaes import *
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import to_json, from_json, StoreSpec
import idaes.core.util.scaling as iscale
from idaes.core.util.model_diagnostics import DegeneracyHunter

import finitevolume
from idaes.core.solvers.homotopy import homotopy
from idaes.core.initialization.block_triangularization import (
    BlockTriangularizationInitializer,
)


# Creating pyomo model
def RPB_model(mode):
    m = ConcreteModel()

    z_bounds = (0, 1)
    # z_init_points = tuple(np.linspace(0.0,0.01,5))+tuple(np.linspace(0.99,1,5))
    z_init_points = tuple(np.linspace(0, 0.1, 10)) + (0.99,)
    # z_init_points = (0.01, 0.99)

    o_bounds = (0, 1)
    # o_init_points = tuple(np.linspace(0.0,0.01,5))+tuple(np.linspace(0.99,1,5))
    o_init_points = tuple(np.linspace(0, 0.25, 10)) + (0.99,)
    # o_init_points = (0.01, 0.99)

    m.z = ContinuousSet(
        doc="axial nodes [dimensionless]",
        bounds=z_bounds,
        initialize=z_init_points,
    )

    m.o = ContinuousSet(
        doc="adsorption theta nodes [dimensionless]",
        bounds=o_bounds,
        initialize=o_init_points,
    )

    disc_method = "Finite Difference"

    if disc_method == "Collocation":
        FiniteElem = 20
        FiniteElem_o = 5
        Collpoints_z = 2
        Collpoints_o = 2
    elif disc_method == "Finite Difference":
        FiniteElem = 20
        FiniteElem_o = 20
    elif disc_method == "Finite Volume":
        FiniteVol = 10
        FiniteVol_o = 10

    # Model Constants

    m.R = Param(
        initialize=8.314e-3,
        units=units.kJ / units.mol / units.K,
        doc="gas constant [kJ/mol/K]",
    )
    m.Rg = Param(
        initialize=8.314e-5,
        units=units.m**3 * units.bar / units.K / units.mol,
        doc="gas constant [m^3*bar/K/mol]",
    )
    m.pi = Param(initialize=3.14159, doc="Pi constant")

    # Initial/Inlet/Outlet Values

    m.component_list = Set(initialize=["N2", "CO2", "H2O"], doc="List of components")

    m.ads_components = Set(
        initialize=["CO2"], within=m.component_list, doc="list of adsorbing components"
    )

    # ========================== Dimensions ========================================
    m.D = Var(initialize=(10), units=units.m, doc="Bed diameter [m]")
    m.D.fix()

    m.L = Var(initialize=(3), units=units.m, doc="Bed Length [m]")
    m.L.fix()

    m.theta = Var(initialize=(0.5), doc="Fraction of bed [-]")
    m.theta.fix()

    m.w_rpm = Var(
        initialize=(1),
        units=units.revolutions / units.min,
        doc="bed rotational speed [revolutions/min]",
    )
    m.w_rpm.fix()

    m.Hx_frac = Param(
        initialize=(1 / 3),  # current assumption, HE takes up 1/3 of bed
        mutable=True,
        doc="fraction of total reactor volume occupied by the embedded"
        "heat exchanger",
    )

    @m.Expression(doc="bed rotational speed [radians/s]")
    def w(m):
        return m.w_rpm * 2 * m.pi / 60

    @m.Expression(doc="cross sectional area, total area*theta [m^2]")
    def A_c(m):
        return m.pi * m.D**2 / 4 * m.theta

    @m.Expression(doc="cross sectional area for flow [m^2]")
    def A_b(m):
        return (1 - m.Hx_frac) * m.A_c  # current assumption, HE takes up 1/3 of bed

    @m.Expression(doc="cross sectional area of the heat exchanger [m^2]")
    def Ahx(m):
        return m.Hx_frac * m.A_c

    @m.Expression(doc="specific heat transfer area [m^2/m^3]")
    def a_ht(m):  # current assumption, can update this later
        return 50

    # ============================ Gas Inlet =======================================
    m.F_in = Var(
        initialize=400,
        doc="Inlet adsorber gas flow [mol/s]",
        bounds=(0, None),
        units=units.mol / units.s,
    )

    m.P_in = Var(
        initialize=1.1,
        bounds=(1, 1.5),
        units=units.bar,
        doc="Inlet flue gas pressure [bar]",
    )

    if mode == "adsorption":
        Tg_in = 90 + 273
        y_in = {"CO2": 0.04, "N2": 0.87, "H2O": 0.09}
    elif mode == "desorption":
        Tg_in = 120 + 273
        y_in = {"CO2": 1e-5, "N2": 1e-3, "H2O": (1 - 1e-5 - 1e-3)}

    m.Tg_in = Var(initialize=Tg_in, units=units.K, doc="Inlet flue gas temperature [K]")

    m.y_in = Var(
        m.component_list,
        initialize=y_in,
        doc="inlet mole fraction",
    )

    # Inlet values for initialization
    @m.Expression(doc="inlet total conc. [mol/m^3]")
    def C_tot_in(m):
        return m.P_in / m.Tg_in / m.Rg

    @m.Expression(m.component_list, doc="inlet concentrations [mol/m^3]")
    def C_in(m, k):
        return m.y_in[k] * m.C_tot_in

    @m.Expression(doc="inlet gas velocity, adsorption [m/s]")
    def vel0(m):
        return m.F_in / m.C_tot_in / m.A_b

    # =========================== Gas Outlet =======================================
    m.P_out = Var(
        initialize=1.01325,
        bounds=(0.99, 1.2),
        units=units.bar,
        doc="Outlet adsorber pressure [bar]",
    )

    m.F_out = Var(
        initialize=m.F_in(),
        bounds=(0, None),
        units=units.mol / units.s,
        doc="Total gas outlet flow [mol/s]",
    )

    m.y_out = Var(
        m.component_list,
        bounds=(0, 1),
        initialize=y_in,
        doc="outlet mole fraction",
    )

    m.Tg_out = Var(
        initialize=100 + 273.15,
        bounds=(75 + 273.15, 150 + 273.15),
        doc="outlet gas temperature [K]",
    )

    # =========================== Solids Inlet =====================================
    if mode == "adsorption":
        qCO2_in = 1
        Ts_in = 100 + 273
    elif mode == "desorption":
        qCO2_in = 2.5
        Ts_in = 120 + 273

    m.qCO2_in = Var(m.z, initialize=qCO2_in, doc="inlet CO2 loading loading [mol/kg]")

    m.Ts_in = Var(m.z, initialize=Ts_in, doc="solids inlet temperature [K]")

    # =========================== Solids Properties ================================
    m.eb = Param(initialize=(0.68), doc="bed voidage")
    m.ep = Param(initialize=(0.68), doc="particle porosity")
    m.dp = Param(initialize=(0.000525), units=units.m, doc="particle diameter [m]")
    m.Cp_sol = Param(
        initialize=(1.457),
        units=units.kJ / units.kg / units.K,
        doc="solid heat capacity [kJ/kg/K]",
    )
    m.rho_sol = Param(
        initialize=(1144),
        units=units.kg / units.m**3,
        mutable=True,
        doc="solid particle densitry [kg/m^3]",
    )

    @m.Expression(doc="particle radius [m]")
    def rp(m):
        return m.dp / 2

    @m.Expression(
        doc="specific particle area for mass transfer, bed voidage"
        "included [m^2/m^3 bed]"
    )
    def a_s(m):
        return 6 / m.dp * (1 - m.eb)

    # ======================== Heat exchanger ======================================
    m.hgx = Param(
        initialize=(25 * 1e-3),  # assumed value
        units=units.kW / units.m**2 / units.K,
        doc="heat exchanger heat transfer coeff. W/m^2/K",
    )

    if mode == "adsorption":
        Tx = 90 + 273
    elif mode == "desorption":
        Tx = 120 + 273

    m.Tx = Var(
        initialize=Tx,
        units=units.K,
        doc="heat exchange fluid temperature, constant [K]",
    )
    m.Tx.fix()

    # Section 1: Chemical and heat exchange properties

    m.MW = Param(
        m.component_list,
        initialize=({"CO2": 44.01e-3, "N2": 28.0134e-3, "H2O": 18.01528e-3}),
        units=units.kg / units.mol,
        doc="component molecular weight [kg/mol]",
    )

    m.mu = Param(
        m.component_list,
        initialize=({"CO2": 1.87e-5, "N2": 2.3e-5, "H2O": 1.26e-5}),
        units=units.Pa * units.s,
        doc="pure component gas phase viscosity [Pa*s]",
    )

    m.k = Param(
        m.component_list,
        initialize=({"CO2": 0.020e-3, "N2": 0.030e-3, "H2O": 0.025e-3}),
        units=units.kW / units.m / units.K,
        doc="pure component gas phase thermal conductivity [kW/m/K]",
    )

    m.DmCO2 = Param(
        initialize=(5.3e-5),
        units=units.m**2 / units.s,
        doc="gas phase CO2 diffusivity [m^2/s]",
    )

    # Variable declaration
    # ============================== Gas Phase =====================================
    m.C = Var(
        m.component_list,
        m.z,
        m.o,
        initialize=1,
        bounds=(1e-8, 100),
        doc="Gas phase Conc. [mol/m^3]",
        units=units.mol / units.m**3,
    )

    m.C_tot = Var(
        m.z,
        m.o,
        initialize=value(m.C_tot_in),
        bounds=(1e-5, 100),
        doc="Total conc., [mol/m^3] (ideal gas law)",
        units=units.mol / units.m**3,
    )

    m.y = Var(
        m.component_list,
        m.z,
        m.o,
        bounds=(1e-10, 1),  # don't need upper bound due to sum(y)=1
        initialize=0.1,
        doc="gas phase mole fraction",
    )

    m.Tg = Var(
        m.z,
        m.o,
        initialize=m.Tg_in.value,
        # domain=PositiveReals,
        bounds=(80 + 273.15, 140 + 273.15),
        doc="Gas phase temperature [K]",
        units=units.K,
    )

    m.heat_flux = Var(
        m.z,
        m.o,
        initialize=0,
        units=units.kJ / units.m**2 / units.s,
        doc="heat flux [kW/m^2 or kJ/s/m^2]",
    )

    m.dheat_fluxdz = DerivativeVar(
        m.heat_flux,
        wrt=m.z,
        doc="axial derivative of heat flux [kW/m^2/dimensionless bed length]",
    )

    # m.dTgdz = DerivativeVar(
    #     m.Tg,
    #     wrt=m.z,
    #     doc="axial derivative of gas phase temp. [K/dimensionless bed length]",
    # )

    m.vel = Var(
        m.z,
        m.o,
        initialize=value(m.vel0),
        # domain=PositiveReals,
        bounds=(0.01, 5),
        doc="gas velocity [m/s], adsorption",
    )

    m.P = Var(
        m.z,
        m.o,
        initialize=m.P_in.value,
        # domain=PositiveReals,
        bounds=(0.99, 1.2),
        units=units.bar,
        doc="Gas Pressure [bar]",
    )

    m.dPdz = DerivativeVar(
        m.P,
        wrt=m.z,
        bounds=(None, 0),
        doc="axial derivative of pressure [bar/dimensionless bed length]",
    )

    m.Flux_kzo = Var(
        m.component_list,
        m.z,
        m.o,
        bounds=(1e-10, 500),
        units=units.mol / units.m**2 / units.s,
        doc="Gas phse component flux [mol/m^2/s]",
    )

    m.dFluxdz = DerivativeVar(
        m.Flux_kzo,
        wrt=m.z,
        doc="axial derivative of component flux [mol/m^2 bed/s/dimensionless bed length]",
    )

    # ========================= Solids =============================================

    m.qCO2 = Var(
        m.z,
        m.o,
        initialize=qCO2_in,
        # within=NonNegativeReals,
        bounds=(1e-5, 5),
        doc="CO2 loading [mol/kg]",
        units=units.mol / units.kg,
    )

    m.dqCO2do = DerivativeVar(m.qCO2, wrt=m.o, doc="theta derivative of loading")

    m.Ts = Var(
        m.z,
        m.o,
        initialize=Ts_in,
        # domain=PositiveReals,
        bounds=(80 + 273.15, 140 + 273.15),
        doc="solid phase temperature [K], adsorption",
        units=units.K,
    )

    m.dTsdo = DerivativeVar(
        m.Ts,
        wrt=m.o,
        doc="theta derivative of solid phase temp. [K/dimensionless bed fraction]",
    )

    # m.Cs_r = Var(
    #     m.z,
    #     m.o,
    #     initialize=value(m.C_in["CO2"]) * 0.8,
    #     # domain=NonNegativeReals,
    #     bounds=(1e-8, 100),
    #     units=units.mol / units.m**3,
    #     doc="particle surface concentration of CO2 [mol/m^3]",
    # )

    # Initialization factors ===
    m.R_MT_gas = Var(
        initialize=1, doc="init. factor for mass transfer in gas phase MB (0=off, 1=on)"
    )
    m.R_MT_gas.fix()

    m.R_MT_solid = Var(
        initialize=1,
        doc="init. factor for mass transfer in solid phase MB (0=off, 1=on)",
    )
    m.R_MT_solid.fix()

    m.R_HT_gs = Var(
        initialize=1, doc="init. factor for gas-to-solid heat transfer (0=off, 1=on)"
    )
    m.R_HT_gs.fix()

    m.R_HT_ghx = Var(
        initialize=1, doc="init. factor for gas-to-HE heat transfer (0=off, 1=on)"
    )
    m.R_HT_ghx.fix()

    m.R_delH = Var(
        initialize=1, doc="init. factor for heat of adsorption (0=off, 1=on)"
    )
    m.R_delH.fix()

    m.R_dP = Var(initialize=1, doc="init. factor for pressure drop (0=off, 1=on)")
    m.R_dP.fix()

    m.R_MT_coeff = Var(
        initialize=1,
        doc="init. factor for the mass transfer coefficient (0=constant value, 1=model prediction)",
    )
    m.R_MT_coeff.fix()

    # Section 2: Gas Equations, Gas Properties, Dimensionless Groups, and Variables related

    @m.Constraint(
        m.z, m.o, doc="total concentration equation (ideal gas law) [mol/m^3]"
    )
    def C_tot_eq(m, z, o):
        return m.C_tot[z, o] * m.Rg * m.Tg[z, o] == m.P[z, o]

    @m.Constraint(
        m.component_list,
        m.z,
        m.o,
        doc="relationship between mole fraction and conc. [-]",
    )
    def y_eq(m, k, z, o):
        return m.y[k, z, o] * m.C_tot[z, o] == m.C[k, z, o]

    @m.Integral(
        m.component_list,
        m.z,
        m.o,
        wrt=m.o,
        doc="Component flow integrated over theta, function of z [mol/s]",
    )
    def Flow_kz(m, k, z, o):
        return m.Flux_kzo[k, z, o] * m.A_b

    @m.Expression(m.z, doc="Total flow integrated over theta, function of z [mol/s]")
    def Flow_z(m, z):
        return sum(m.Flow_kz[k, z] for k in m.component_list)

    @m.Expression(
        m.component_list,
        m.z,
        doc="Component mole fraction integrated over theta, function of z [-]",
    )
    def y_kz(m, k, z):
        return m.Flow_kz[k, z] / m.Flow_z[z]

    @m.Expression(m.z, m.o, doc="Total flux indexed over z and o [mol/s]")
    def Flux_zo(m, z, o):
        return sum(m.Flux_kzo[k, z, o] for k in m.component_list)

    # ======================== Gas properties ======================================
    def Cp_g_(k, Tg):
        if k == "H2O":
            return (
                30.09200
                + 6.832514 * Tg / 1000
                + 6.793435 * (Tg / 1000) ** 2
                + -2.534480 * (Tg / 1000) ** 3
                + 0.082139 / (Tg / 1000) ** 2
            ) / 1000
        elif k == "N2":
            return (
                28.98641
                + 1.853978 * Tg / 1000
                + -9.647459 * (Tg / 1000) ** 2
                + 16.63537 * (Tg / 1000) ** 3
                + 0.000117 / (Tg / 1000) ** 2
            ) / 1000
        elif k == "CO2":
            return (
                24.99735
                + 55.18696 * Tg / 1000
                + -33.69137 * (Tg / 1000) ** 2
                + 7.948387 * (Tg / 1000) ** 3
                + -0.136638 / (Tg / 1000) ** 2
            ) / 1000

    @m.Expression(
        m.component_list,
        m.z,
        m.o,
        doc="pure component heat capacities, function of T [kJ/mol/K]",
    )
    def Cp_g(m, k, z, o):
        return Cp_g_(k, m.Tg[z, o]) * units.kJ / units.mol / units.K

    @m.Expression(m.z, m.o, doc="average molecular weight [kg/mol]")
    def AMW(m, z, o):
        return sum([m.y[k, z, o] * m.MW[k] for k in m.component_list])

    @m.Expression(m.z, m.o, doc="gas density [kg/m^3]")
    def rhog(m, z, o):
        return m.AMW[z, o] * m.C_tot[z, o]

    @m.Expression(m.z, m.o, doc="gas phase mixture heat capacity [kJ/mol/K]")
    def Cp_g_mix(m, z, o):
        return sum([m.y[k, z, o] * m.Cp_g[k, z, o] for k in m.component_list])

    @m.Expression(m.z, m.o, doc="gas phase mixture heat capacity [kJ/kg/K]")
    def Cp_g_mix_kg(m, z, o):
        return m.AMW[z, o] * m.Cp_g_mix[z, o]

    @m.Expression(m.z, m.o, doc="gas phase mixture viscosity [Pa*s]")
    def mu_mix(m, z, o):
        return sum(
            [m.y[k, z, o] * m.mu[k] * m.MW[k] ** 0.5 for k in m.component_list]
        ) / sum([m.y[k, z, o] * m.MW[k] ** 0.5 for k in m.component_list])

    @m.Expression(m.z, m.o, doc="gas mixture thermal conductivity [kW/m/K]")
    def k_mix(m, z, o):
        return sum([m.y[k, z, o] * m.k[k] for k in m.component_list])

    @m.Constraint(m.z, m.o, doc="heat flux equation [kJ/s/m^2]")
    def heat_flux_eq(m, z, o):
        return (
            m.heat_flux[z, o]
            == m.C_tot[z, o] * m.Cp_g_mix[z, o] * m.vel[z, o] * m.Tg[z, o]
        )

    # Dimensionless groups ====

    m.ln_Pr = Var(
        m.z,
        m.o,
        initialize=0,
        # domain=PositiveReals,
        bounds=(-25, 5),
        doc="Prandtl number",
    )
    m.ln_Re = Var(
        m.z,
        m.o,
        initialize=0,
        # domain=PositiveReals,
        bounds=(-5, 5),
        doc="Reynolds number",
    )
    m.ln_Sc = Var(
        m.z,
        m.o,
        initialize=0,
        # domain=PositiveReals,
        bounds=(-5, 5),
        doc="Schmidt number",
    )

    @m.Constraint(m.z, m.o, doc="Prandtl number")
    def Pr_eq(m, z, o):
        return (
            exp(m.ln_Pr[z, o]) * m.k_mix[z, o] == m.mu_mix[z, o] * m.Cp_g_mix_kg[z, o]
        )

    @m.Constraint(m.z, m.o, doc="Reynolds number")
    def Re_eq(m, z, o):
        return exp(m.ln_Re[z, o]) * m.mu_mix[z, o] == m.rhog[z, o] * m.vel[z, o] * m.dp

    @m.Constraint(m.z, m.o, doc="Schmidt number")
    def Sc_eq(m, z, o):
        return exp(m.ln_Sc[z, o]) * m.rhog[z, o] * m.DmCO2 == m.mu_mix[z, o]

    @m.Expression(m.z, m.o, doc="Sherwood number")
    def Sh(m, z, o):
        return 2.0 + 0.6 * exp(0.33 * m.ln_Re[z, o] + 0.5 * m.ln_Sc[z, o])

    @m.Expression(m.z, m.o, doc="Nusselt number")
    def Nu(m, z, o):
        return 2.0 + 1.1 * exp(0.6 * m.ln_Re[z, o] + 0.33 * m.ln_Pr[z, o])

    # ===

    # Mass/Heat Transfer variables

    # m.h_gs = Var(
    #     m.z,
    #     m.o,
    #     initialize=0.111,
    #     bounds=(0.005, 0.5),
    #     doc="Gas-solid heat transfer coefficient [kW/m^2/K]",
    # )

    @m.Expression(
        m.z, m.o, doc="Gas-solid heat transfer coefficient  equation [kW/m^2/K]"
    )
    def h_gs(m, z, o):
        # return m.h_gs[z, o] == m.Nu[z, o] * m.k_mix[z, o] / m.dp
        return m.Nu[z, o] * m.k_mix[z, o] / m.dp

    # @m.Expression(m.z, m.o, doc="Gas phase film mass transfer coefficient [m/s]")
    # def k_f(m, z, o):
    #     return m.Sh[z, o] * m.DmCO2 / m.dp

    # ===

    # Isotherm model =======
    # Parameter values
    q_inf_1 = 2.87e-02  # mmol/g == mol/kg
    q_inf_2 = 1.95
    q_inf_3 = 3.45

    d_inf_1 = 1670.31
    d_inf_2 = 789.01
    d_inf_3 = 10990.67
    d_inf_4 = 0.28

    E_1 = -76.15
    E_2 = -77.44
    E_3 = -194.48
    E_4 = -6.76

    X_11 = 4.20e-02
    X_21 = 2.97
    X_12 = 7.74e-02
    X_22 = 1.66

    P_step_01 = 1.85e-03
    P_step_02 = 1.78e-02
    ln_P0_1 = -6.2925
    ln_P0_2 = -4.0285

    H_step_1 = -99.64
    H_step_2 = -78.19

    gamma_1 = 894.67
    gamma_2 = 95.22

    T0 = 363.15

    # Model equations
    def d_1(T):
        return d_inf_1 * exp(-E_1 / (m.R * T0) * (T0 / T - 1))

    def d_2(T):
        return d_inf_2 * exp(-E_2 / (m.R * T0) * (T0 / T - 1))

    def d_3(T):
        return d_inf_3 * exp(-E_3 / (m.R * T0) * (T0 / T - 1))

    def d_4(T):
        return d_inf_4 * exp(-E_4 / (m.R * T0) * (T0 / T - 1))

    def sigma_1(T):
        return X_11 * exp(X_21 * (1 / T0 - 1 / T))

    def sigma_2(T):
        return X_12 * exp(X_22 * (1 / T0 - 1 / T))

    def ln_pstep1(T):
        return ln_P0_1 + (-H_step_1 / m.R * (1 / T0 - 1 / T))

    def ln_pstep2(T):
        return ln_P0_2 + (-H_step_2 / m.R * (1 / T0 - 1 / T))

    def q_star_1(P, T):
        return q_inf_1 * d_1(T) * P / (1 + d_1(T) * P)

    def q_star_2(P, T):
        return q_inf_2 * d_2(T) * P / (1 + d_2(T) * P)

    def q_star_3(P, T):
        return q_inf_3 * d_3(T) * P / (1 + d_3(T) * P) + d_4(T) * P

    # =============================================

    @m.Expression(
        m.z,
        m.o,
        doc="Partial pressure of CO2 at particle surface [bar] (ideal gas law)",
    )
    def P_surf(m, z, o):
        # return m.Cs_r[z, o] * m.Rg * m.Ts[z, o]
        return m.P[z, o] * m.y["CO2", z, o]

    @m.Expression(m.z, m.o, doc="log(Psurf)")
    def ln_Psurf(m, z, o):
        return log(m.P_surf[z, o])

    @m.Expression(m.z, m.o, doc="weighting function term1: (ln_Psurf-ln_Pstep)/sigma")
    def iso_w_term1(m, z, o):
        return (m.ln_Psurf[z, o] - ln_pstep1(m.Ts[z, o])) / sigma_1(m.Ts[z, o])

    @m.Expression(m.z, m.o, doc="weighting function term2: (ln_Psurf-ln_Pstep)/sigma")
    def iso_w_term2(m, z, o):
        return (m.ln_Psurf[z, o] - ln_pstep2(m.Ts[z, o])) / sigma_2(m.Ts[z, o])

    @m.Expression(m.z, m.o, doc="log of weighting function 1")
    def ln_w1(m, z, o):
        # return gamma_1*log(exp(m.iso_w_term1[z,o])/(1+exp(m.iso_w_term1[z,o])))
        # return gamma_1*(log(exp(m.iso_w_term1[z,o])) - log(1+exp(m.iso_w_term1[z,o])))
        return gamma_1 * (m.iso_w_term1[z, o] - log(1 + exp(m.iso_w_term1[z, o])))

    @m.Expression(m.z, m.o, doc="log of weighting function 2")
    def ln_w2(m, z, o):
        # return gamma_2*log(exp(m.iso_w_term2[z,o])/(1+exp(m.iso_w_term2[z,o])))
        # return gamma_2*(log(exp(m.iso_w_term2[z,o])) - log(1+exp(m.iso_w_term2[z,o])))
        return gamma_2 * (m.iso_w_term2[z, o] - log(1 + exp(m.iso_w_term2[z, o])))

    @m.Expression(m.z, m.o, doc="weighting function 1")
    def iso_w1(m, z, o):
        return exp(m.ln_w1[z, o])

    @m.Expression(m.z, m.o, doc="weighting function 2")
    def iso_w2(m, z, o):
        return exp(m.ln_w2[z, o])

    @m.Expression(m.z, m.o, doc="isotherm loading expression [mol/kg]")
    def qCO2_eq(m, z, o):
        return (
            (1 - m.iso_w1[z, o]) * q_star_1(m.P_surf[z, o], m.Ts[z, o])
            + (m.iso_w1[z, o] - m.iso_w2[z, o]) * q_star_2(m.P_surf[z, o], m.Ts[z, o])
            + m.iso_w2[z, o] * q_star_3(m.P_surf[z, o], m.Ts[z, o])
        )

    # ======

    # internal mass transfer ===
    m.C1 = Param(initialize=(2.562434e-12), doc="lumped MT parameter [m^2/K^0.5/s]")

    @m.Expression(m.z, m.o, doc="effective diffusion in solids [m^2/s]")
    def Deff(m, z, o):
        return m.C1 * m.Ts[z, o] ** 0.5
        # return m.C1 * exp(0.5 * log(m.Ts[z, o]))

    @m.Expression(m.z, m.o, doc="internal MT coeff. [1/s]")
    def k_I(m, z, o):
        if z < 0.1:
            return 1 * (
                m.R_MT_coeff * (15 * m.ep * m.Deff[z, o] / m.rp**2)
                + (1 - m.R_MT_coeff) * 0.001
            )
        else:
            return (
                m.R_MT_coeff * (15 * m.ep * m.Deff[z, o] / m.rp**2)
                + (1 - m.R_MT_coeff) * 0.001
            )

    # ========

    # axial dispersion coefficient not needed at this time
    # @m.Expression(m.z, m.o, doc="axial dispersion coefficient [?]")
    # def Dz(m, z, o):
    #     return m.DmCO2 / m.eb * (20 + 0.5 * m.Sc[z, o] * m.Re[z, o])

    # heat of adsorption ===

    delH_a1 = 21.68
    delH_a2 = 29.10
    delH_b1 = 1.59
    delH_b2 = 3.39
    delH_1 = 98.76
    delH_2 = 77.11
    delH_3 = 21.25

    @m.Expression(m.z, m.o, doc="heat of adsorption [kJ/mol]")
    def delH_CO2(m, z, o):
        return -(delH_1 - (delH_1 - delH_2)) * exp(
            delH_a1 * (m.qCO2_eq[z, o] - delH_b1)
        ) / (1 + exp(delH_a1 * (m.qCO2_eq[z, o] - delH_b1))) - (delH_2 - delH_3) * exp(
            delH_a2 * (m.qCO2_eq[z, o] - delH_b2)
        ) / (
            1 + exp(delH_a2 * (m.qCO2_eq[z, o] - delH_b2))
        )

    # ===

    # mass transfer rates ===
    @m.Expression(m.z, m.o, doc="solids mass transfer rate [mol/s/m^3 bed]")
    def Rs_CO2(m, z, o):
        if 0 < z < 1 and 0 < o < 1:  # no mass transfer at boundaries
            return (
                m.k_I[z, o] * (m.qCO2_eq[z, o] - m.qCO2[z, o]) * (1 - m.eb) * m.rho_sol
            )
        else:
            return 0

    @m.Expression(m.z, m.o, doc="gas mass transfer rate [mol/s/m^3 bed]")
    def Rg_CO2(m, z, o):
        if 0 < z < 1 and 0 < o < 1:  # no mass transfer at boundaries
            # return m.k_f[z, o] * m.a_s * (m.C["CO2", z, o] - m.Cs_r[z, o])  # option 1
            return m.Rs_CO2[z, o]  # option 2
        else:
            return 0

    # heat transfer rates ===
    @m.Expression(
        m.z, m.o, doc="Gas-to-solid heat transfer rate [kW/m^3 bed or kJ/s/m^3 bed]"
    )
    def Q_gs(m, z, o):
        if 0 < z < 1 and 0 < o < 1:  # no heat transfer at boundaries
            return m.R_HT_gs * m.h_gs[z, o] * m.a_s * (m.Ts[z, o] - m.Tg[z, o])
        else:
            return 0

    @m.Expression(
        m.z, m.o, doc="Gas-to-HX heat transfer rate [kW/m^3 bed or kJ/s/m^3 bed]"
    )
    def Q_ghx(m, z, o):
        if 0 < z < 1 and 0 < o < 1:  # no heat transfer at boundaries
            return m.R_HT_ghx * m.hgx * m.a_ht * (m.Tg[z, o] - m.Tx)
        else:
            return 0

    @m.Expression(m.z, m.o, doc="adsorption/desorption heat rate [kJ/s/m^3 bed]")
    def Q_delH(m, z, o):
        return m.R_delH * m.delH_CO2[z, o] * m.Rs_CO2[z, o]

    # PDE equations, boundary conditions, and model constraints
    @m.Constraint(
        m.component_list, m.z, m.o, doc="gas phase species balance PDE [mol/m^3 bed/s]"
    )
    def pde_gasMB(m, k, z, o):
        if 0 < z < 1:
            if k == "CO2":
                return m.dFluxdz[k, z, o] / m.L == -m.Rg_CO2[z, o] * m.R_MT_gas
            else:
                return m.dFluxdz[k, z, o] / m.L == 0
        if z == 1:  # at exit of column, dFluxdz=0
            return m.dFluxdz[k, z, o] == 0
        else:  # no balance at z=0, inlets are specified
            return Constraint.Skip

    @m.Constraint(m.component_list, m.z, m.o, doc="flux equation [mol/m^2 bed/s]")
    def flux_eq(m, k, z, o):
        return m.Flux_kzo[k, z, o] == m.C[k, z, o] * m.vel[z, o]

    @m.Constraint(m.z, m.o, doc="solid phase mass balance PDE [mol/m^3 bed/s]")
    def pde_solidMB(m, z, o):
        if 0 < o < 1:
            return (1 - m.eb) * m.rho_sol * m.dqCO2do[z, o] * m.w / (
                2 * m.pi * m.theta
            ) == m.Rs_CO2[z, o] * m.R_MT_solid
        elif o == 1:  # at solids exit, flux is zero
            return m.dqCO2do[z, o] == 0
        else:  # no balance at o=0, inlets are specified
            return Constraint.Skip

    @m.Constraint(m.z, m.o, doc="gas phase energy balance PDE [kJ/m^3 bed/s]")
    def pde_gasEB(m, z, o):
        if 0 < z < 1:
            return m.dheat_fluxdz[z, o] / m.L == m.Q_gs[z, o] - m.Q_ghx[z, o]
        elif z == 1:
            return m.dheat_fluxdz[z, o] == 0
        else:
            return Constraint.Skip

    @m.Constraint(m.z, m.o, doc="solid phase energy balance PDE [kJ/s/m^3 bed]")
    def pde_solidEB(m, z, o):
        if 0 < o < 1:
            return (1 - m.eb) * m.rho_sol * m.Cp_sol * m.w * m.dTsdo[z, o] / (
                2 * m.pi * m.theta
            ) == -m.Q_gs[z, o] - m.Q_delH[z, o]
        elif o == 1:
            return m.dTsdo[z, o] == 0
        else:
            return Constraint.Skip

    # @m.Constraint(m.z, m.o, doc="gas and solid phase mass transfer continuity")
    # def constr_MTcont(m, z, o):
    #     """
    #     Mass transfer continuity between the gas and solid phase. Used to calculate
    #     Csurf which sets driving force for gas phase mass transfer. A couple options
    #     for how to write this.

    #     If m.Rg_CO2 = m.kf*m.a_s*(m.C['CO2']-m.Cs_r) set as expression, then:
    #         m.Rg_CO2[z,o] == m.Rs_CO2[z,o]

    #     If m.Rg_CO2 = m.Rs_CO2 set as expression, then a couple options:
    #         1) m.Rg_CO2[z,o] == m.k_f[z,o]*m.a_s*(m.C['CO2',z,o]-m.Cs_r[z,o])
    #         2) m.C['CO2',z,o] == m.Rg_CO2[z,o]/m.k_f[z,o]/m.a_s + m.Cs_r[z,o]

    #     """
    #     return (
    #         m.Cs_r[z, o] == m.C["CO2", z, o] - m.Rg_CO2[z, o] / m.k_f[z, o] / m.a_s
    #     )  # option 2b
    #     # if 0 < z < 1 and 0 < o < 1:
    #     #     return m.Rg_CO2[z, o] == m.Rs_CO2[z, o]
    #     # else:
    #     #     return m.Cs_r[z, o] == m.C["CO2", z, o]

    @m.Constraint(m.z, m.o, doc="Ergun Equation [bar/m]")
    def pde_Ergun(m, z, o):
        if z > 0:
            return m.dPdz[z, o] / m.L == m.R_dP * -(
                1e-5
                * (150 * m.mu_mix[z, o] * ((1 - m.eb) ** 2) / (m.eb**3))
                / m.dp**2
                * m.vel[z, o]
                + 1e-5
                * 1.75
                * (1 - m.eb)
                / m.eb**3
                * m.rhog[z, o]
                / m.dp
                * m.vel[z, o] ** 2
            )
        else:
            return Constraint.Skip

    @m.Constraint(m.z, m.o, doc="mole fraction summation")
    def mole_frac_sum(m, z, o):
        if z > 0:
            return sum([m.y[k, z, o] for k in m.component_list]) == 1
        else:
            return Constraint.Skip

    # Boundary Conditions ===

    @m.Constraint(m.z, doc="inlet solids loading B.C. [mol/kg]")
    def bc_q_in(m, z):
        return m.qCO2[z, 0] == m.qCO2_in[z]

    @m.Constraint(m.o, doc="inlet gas temp. B.C. [K]")
    def bc_gastemp_in(m, o):
        return m.Tg[0, o] == m.Tg_in

    @m.Constraint(m.z, doc="inlet solids temp. [K]")
    def bc_solidtemp_in(m, z):
        return m.Ts[z, 0] == m.Ts_in[z]

    @m.Constraint(m.o, doc="inlet pressure [bar]")
    def bc_P_in(m, o):
        return m.P[0, o] == m.P_in

    @m.Constraint(doc="inlet flow B.C. [mol/s]")
    def bc_flow_in(m):
        return m.Flow_z[0] == m.F_in

    @m.Constraint(m.component_list, m.o, doc="inlet mole fraction B.C. [-]")
    def bc_y_in(m, k, o):
        return m.y[k, 0, o] == m.y_in[k]

    # Outlet values ==========

    @m.Integral(m.o, wrt=m.o, doc="outlet gas enthalpy [kJ/mol]")
    def Hg_out(m, o):
        return (
            sum(m.Flux_kzo[k, 1, o] for k in m.component_list)
            * m.Cp_g_mix[1, o]
            * m.Tg[1, o]
            * m.A_b
        )

    @m.Expression(doc="outlet gas heat capacity [kJ/mol/K]")
    def Cp_g_out(m):
        return sum([m.y_out[k] * Cp_g_(k, m.Tg_out) for k in m.component_list])

    @m.Constraint(doc="eq. for calculating outlet gas temperature")
    def Tg_out_eq(m):
        return m.Hg_out == m.F_out * m.Cp_g_out * m.Tg_out

    @m.Constraint(doc="Outlet flow B.C.")
    def bc_flow_out(m):
        return m.F_out == m.Flow_z[1]

    @m.Constraint(m.component_list, doc="Outlet mole fraction B.C.")
    def bc_y_out(m, k):
        return m.y_out[k] == m.y_kz[k, 1]

    @m.Constraint(m.o, doc="outlet pressure B.C. [bar]")
    def bc_P_out(m, o):
        return m.P[1, o] == m.P_out

    # Metrics ==============
    m.CO2_capture = Var(initialize=0.5, doc="CO2 capture fraction")

    @m.Constraint(doc="CO2 capture fraction")
    def CO2_capture_eq(m):
        # return 1 - (m.F_out * m.y_out["CO2"]) / (m.F_in * m.y_in["CO2"])
        return m.CO2_capture * (m.F_in * m.y_in["CO2"]) == (m.F_in * m.y_in["CO2"]) - (
            m.F_out * m.y_out["CO2"]
        )

    # ==============

    # Mass and energy balance checks

    @m.Expression(m.component_list, doc="component gas flow in for MB check [mol/s]")
    def g_k_in_MB(m, k):
        return m.F_in * m.y_in[k]

    @m.Expression(m.component_list, doc="gas flow out for MB check [mol/s]")
    def g_k_out_MB(m, k):
        return m.F_out * m.y_out[k]

    @m.Expression(doc="total bed volume [m^3]")
    def vol_tot(m):
        return m.pi * (m.D / 2) ** 2 * m.L * (1 - m.Hx_frac)

    @m.Expression(doc="total solids volume [m^3]")
    def vol_solids_tot(m):
        return m.vol_tot * (1 - m.eb)

    @m.Expression(doc="total solids mass [kg]")
    def mass_solids_tot(m):
        return m.vol_solids_tot * m.rho_sol

    @m.Expression(doc="total solids flow [kg/s]")
    def flow_solids_tot(m):
        return m.mass_solids_tot * m.w_rpm / 60

    @m.Expression(m.z, doc="change in solids loading at each z index [mol/kg]")
    def delta_q(m, z):
        return m.qCO2[z, 1] - m.qCO2[z, 0]

    @m.Integral(m.z, wrt=m.z, doc="total flow of adsorbed CO2 at inlet [mol/s]")
    def flow_CO2solids_in(m, z):
        return m.qCO2[z, 0] * m.flow_solids_tot

    @m.Integral(m.z, wrt=m.z, doc="total flow of adsorbed CO2 at outlet [mol/s]")
    def flow_CO2solids_out(m, z):
        return m.qCO2[z, 1] * m.flow_solids_tot

    @m.Expression(m.component_list, doc="% mass balance error for each component")
    def MB_error(m, k):
        if k == "CO2":
            return (
                (m.flow_CO2solids_out + m.g_k_out_MB["CO2"])
                / (m.flow_CO2solids_in + m.g_k_in_MB["CO2"])
                - 1
            ) * 100
        else:
            return (
                m.g_k_out_MB[k] / m.g_k_in_MB[k] - 1
            ) * 100  # (out-in)/in*100 or (out/in-1)*100

    # DAE Transformations

    if disc_method == "Collocation":
        discretizer = TransformationFactory("dae.collocation")
        discretizer.apply_to(m, wrt=m.z, nfe=FiniteElem, ncp=Collpoints_z)
        discretizer.apply_to(m, wrt=m.o, nfe=FiniteElem_o, ncp=Collpoints_o)
    elif disc_method == "Finite Difference":
        discretizer = TransformationFactory("dae.finite_difference")
        discretizer.apply_to(m, wrt=m.z, nfe=FiniteElem)
        discretizer.apply_to(m, wrt=m.o, nfe=FiniteElem_o)
    elif disc_method == "Finite Volume":
        discretizer = TransformationFactory("dae.finite_volume")
        discretizer.apply_to(
            m, wrt=m.z, nfv=FiniteVol, scheme="Minmod", flow_direction=1
        )
        discretizer.apply_to(
            m, wrt=m.o, nfv=FiniteVol_o, scheme="Minmod", flow_direction=1
        )

    # initializing some variables and setting scaling factors

    # need to do this after discretizer is applied or else the new indices won't get the scaling factor

    # initializing variables
    for z in m.z:
        for o in m.o:
            for k in m.component_list:
                m.C[k, z, o] = value(m.C_in[k])
                m.y[k, z, o] = value(m.C[k, z, o] / m.C_tot[z, o])
                m.Flux_kzo[k, z, o] = value(m.C[k, z, o] * m.vel[z, o])

    # scaling factors ================================
    iscale.set_scaling_factor(m.bc_P_in, 10)
    iscale.set_scaling_factor(m.bc_y_out["CO2"], 100)
    iscale.set_scaling_factor(m.bc_y_out["H2O"], 1 / value(m.y_in["H2O"]))
    iscale.set_scaling_factor(m.bc_y_out["N2"], 1 / value(m.y_in["N2"]))
    iscale.set_scaling_factor(m.bc_P_out, 10)
    iscale.set_scaling_factor(m.Tg_out_eq, 1e-3)
    iscale.set_scaling_factor(m.Tg_out, 1e-2)
    iscale.set_scaling_factor(m.y_out["H2O"], 1 / value(m.y_in["H2O"]))
    iscale.set_scaling_factor(m.y_out["N2"], 1 / value(m.y_in["N2"]))
    iscale.set_scaling_factor(m.y_out["CO2"], 100)

    for z in m.z:
        iscale.set_scaling_factor(m.bc_solidtemp_in[z], 1e-2)
        iscale.set_scaling_factor(m.bc_q_in[z], 10)
        for o in m.o:
            iscale.set_scaling_factor(m.vel[z, o], 10)
            iscale.set_scaling_factor(m.qCO2[z, o], 10)
            iscale.set_scaling_factor(m.Tg[z, o], 1e-2)
            iscale.set_scaling_factor(m.Ts[z, o], 1e-2)
            iscale.set_scaling_factor(m.P[z, o], 10)
            iscale.set_scaling_factor(m.Pr_eq[z, o], 1e8)
            iscale.set_scaling_factor(m.Re_eq[z, o], 1e4)
            iscale.set_scaling_factor(m.flux_eq["CO2", z, o], 10)
            iscale.set_scaling_factor(m.flux_eq["H2O", z, o], 10)
            iscale.set_scaling_factor(m.Sc_eq[z, o], 1e5)
            iscale.set_scaling_factor(m.heat_flux_eq[z, o], 1e-2)
            iscale.set_scaling_factor(m.y["H2O", z, o], 1 / value(m.y_in["H2O"]))
            iscale.set_scaling_factor(m.y["N2", z, o], 1 / value(m.y_in["N2"]))
            iscale.set_scaling_factor(m.C_tot_eq[z, o], 10)

            if z > 0:
                iscale.set_scaling_factor(m.dFluxdz_disc_eq["CO2", z, o], 0.5)
                iscale.set_scaling_factor(m.dFluxdz_disc_eq["H2O", z, o], 0.5)
                iscale.set_scaling_factor(m.dFluxdz_disc_eq["N2", z, o], 0.1)
                iscale.set_scaling_factor(m.dPdz[z, o], 100)
                iscale.set_scaling_factor(m.dPdz_disc_eq[z, o], 0.5)
                iscale.set_scaling_factor(m.pde_Ergun[z, o], 100)
                iscale.set_scaling_factor(m.dheat_fluxdz_disc_eq[z, o], 1e-2)

            if o > 0:
                iscale.set_scaling_factor(m.dTsdo_disc_eq[z, o], 1e-4)
                iscale.set_scaling_factor(m.dqCO2do_disc_eq[z, o], 0.5)

            if 0 < z < 1 and 0 < o < 1:
                iscale.set_scaling_factor(m.pde_gasEB[z, o], 1e-4)
                iscale.set_scaling_factor(m.pde_solidEB[z, o], 1e-4)
                iscale.set_scaling_factor(m.pde_solidMB[z, o], 1e-1)
                iscale.set_scaling_factor(m.dheat_fluxdz[z, o], 1e-2)
                iscale.set_scaling_factor(m.dTsdo[z, o], 1e-3)
                iscale.set_scaling_factor(m.pde_gasMB["CO2", z, o], 0.1)

            if z == 0:
                iscale.set_scaling_factor(m.y["CO2", z, o], 1 / value(m.y_in["CO2"]))

    for o in m.o:
        iscale.set_scaling_factor(m.bc_gastemp_in[o], 1e-2)
        iscale.set_scaling_factor(m.bc_y_in["CO2", o], 1 / value(y_in["CO2"]))
        iscale.set_scaling_factor(m.bc_y_in["H2O", o], 1 / value(y_in["H2O"]))
        iscale.set_scaling_factor(m.bc_y_in["N2", o], 1 / value(y_in["N2"]))

    # =================================================

    # Setting Initialization factors
    m.R_HT_gs = 1e-10
    m.R_HT_ghx = 1e-10
    m.R_delH = 1e-10
    m.R_MT_coeff = 1e-10
    m.R_dP = 1
    m.R_MT_gas = 1e-10
    m.R_MT_solid = 1e-10

    # fixing solid inlet variables
    for z in m.z:
        m.Ts_in[z].fix()
        m.qCO2_in[z].fix()

    return m


def plotting(blk):
    def find_closest_ind(ind_list, query_values):
        closest_ind = []
        for j in query_values:
            closest_j = min(ind_list, key=lambda x: abs(x - j))
            closest_ind.append(ind_list.index(closest_j))

        return closest_ind

    theta_query = [0.05, 0.5, 0.95]
    z_query = [0.05, 0.5, 0.95]

    def model_plot_CO2g_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        y_CO2 = [[], [], []]
        # theta_query=[0.05,0.5,0.95]
        theta_test = find_closest_ind(theta, theta_query)
        k = 0
        for j in theta_test:
            for i in z:
                y_CO2[k].append(m.y["CO2", i, theta[j]]())
            k += 1

        # fig = plt.figure(figsize=(10,6))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Normalized Axial distance", fontsize=16)
        ax.set_ylabel("Gas phase CO$_{2}$ mole fraction", fontsize=16)
        ax.set_ylim([0, 0.05])
        # ax.set_title('Adsorption gas phase CO$_{2}$')
        for i in range(len(theta_test)):
            ax.plot(z, y_CO2[i], "-o", label="theta=" + str(theta[theta_test[i]]))
        ax.legend()

    model_plot_CO2g_RKH(blk)

    def model_plot_CO2g_conc_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        C_CO2 = [[], [], []]
        # theta_query=[0.05,0.5,0.95]
        theta_test = find_closest_ind(theta, theta_query)
        k = 0
        for j in theta_test:
            for i in z:
                C_CO2[k].append(m.C["CO2", i, theta[j]]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Normalized Axial distance", fontsize=16)
        ax.set_ylabel("Gas phase CO$_{2}$ conc.", fontsize=16)
        # ax.set_ylim([0,0.05])
        # ax.set_title('Adsorption gas phase CO$_{2}$')
        for i in range(len(theta_test)):
            ax.plot(z, C_CO2[i], "-o", label="theta=" + str(theta[theta_test[i]]))
        ax.legend()

    model_plot_CO2g_conc_RKH(blk)

    def model_plot_N2g_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        C_N2 = [[], [], []]
        # theta_query=[0.05,0.5,0.95]
        theta_test = find_closest_ind(theta, theta_query)
        k = 0
        for j in theta_test:
            for i in z:
                C_N2[k].append(m.C["N2", i, theta[j]]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Normalized Axial distance", fontsize=16)
        ax.set_ylabel("Gas phase N$_{2}$ conc.", fontsize=16)
        # ax.set_ylim([0,0.05])
        # ax.set_title('Adsorption gas phase CO$_{2}$')
        for i in range(len(theta_test)):
            ax.plot(z, C_N2[i], "-o", label="theta=" + str(theta[theta_test[i]]))
        ax.legend()

    model_plot_N2g_RKH(blk)

    def model_plot_Tg_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        Tg = [[], [], []]
        # theta_query=[0.05,0.5,0.95]
        theta_test = find_closest_ind(theta, theta_query)
        k = 0
        for j in theta_test:
            for i in z:
                Tg[k].append(m.Tg[i, theta[j]]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Normalized Axial distance", fontsize=16)
        ax.set_ylabel("Gas Temperature [K]", fontsize=16)
        # ax.set_title('Adsorption gas phase CO$_{2}$')
        for i in range(len(theta_test)):
            ax.plot(z, Tg[i], "-o", label="theta=" + str(theta[theta_test[i]]))
        ax.legend()

    model_plot_Tg_RKH(blk)

    def model_plot_Pg_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        Pg = [[], [], []]
        # theta_query=[0.05,0.5,0.95]
        theta_test = find_closest_ind(theta, theta_query)
        k = 0
        for j in theta_test:
            for i in z:
                Pg[k].append(m.P[i, theta[j]]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Normalized Axial distance", fontsize=16)
        ax.set_ylabel("Gas Pressure [bar]", fontsize=16)
        # ax.set_title('Adsorption gas phase CO$_{2}$')
        for i in range(len(theta_test)):
            ax.plot(z, Pg[i], "-o", label="theta=" + str(theta[theta_test[i]]))
        ax.legend()

    model_plot_Pg_RKH(blk)

    def model_plot_vg_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        vg = [[], [], []]
        # theta_query=[0.05,0.5,0.95]
        theta_test = find_closest_ind(theta, theta_query)
        k = 0
        for j in theta_test:
            for i in z:
                vg[k].append(m.vel[i, theta[j]]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Normalized Axial distance", fontsize=16)
        ax.set_ylabel("Gas velocity [m/s]", fontsize=16)
        # ax.set_title('Adsorption gas phase CO$_{2}$')
        for i in range(len(theta_test)):
            ax.plot(z, vg[i], "-o", label="theta=" + str(theta[theta_test[i]]))
        ax.legend()

    model_plot_vg_RKH(blk)

    def model_plot_CO2s_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        qCO2 = [[], [], []]
        # z_query=[0.05,0.5,0.95]
        z_nodes = find_closest_ind(z, z_query)
        k = 0
        for j in z_nodes:
            for i in theta:
                qCO2[k].append(m.qCO2[z[j], i]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Theta distance (radians)", fontsize=16)
        ax.set_ylabel("CO$_{2}$ Loading [mol/kg]", fontsize=16)
        # ax.set_title('Adsorption CO$_{2}$ Loading')
        for i in range(len(z_nodes)):
            ax.plot(theta, qCO2[i], "-o", label="z=" + str(z[z_nodes[i]]))
        ax.legend()

    model_plot_CO2s_RKH(blk)

    def model_plot_Ts_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        Ts = [[], [], []]
        # z_query=[0.05,0.5,0.95]
        z_nodes = find_closest_ind(z, z_query)
        k = 0
        for j in z_nodes:
            for i in theta:
                Ts[k].append(m.Ts[z[j], i]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Theta distance (radians)", fontsize=16)
        ax.set_ylabel("Solids Temperature [K]", fontsize=16)
        # ax.set_title('Adsorption CO$_{2}$ Loading')
        for i in range(len(z_nodes)):
            ax.plot(theta, Ts[i], "-o", label="z=" + str(z[z_nodes[i]]))
        ax.legend()

    model_plot_Ts_RKH(blk)

    def model_plot_yCO2theta_RKH(m):
        z = list(m.z)
        theta = list(m.o)
        y = [[], [], []]
        # z_query=[0.05,0.5,0.95]
        z_nodes = find_closest_ind(z, z_query)
        k = 0
        for j in z_nodes:
            for i in theta:
                y[k].append(m.y["CO2", z[j], i]())
            k += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Theta distance (radians)", fontsize=16)
        ax.set_ylabel("CO$_{2}$ mole fraction", fontsize=16)
        ax.set_ylim([0, 0.05])
        # ax.set_title('Adsorption CO$_{2}$ Loading')
        for i in range(len(z_nodes)):
            ax.plot(theta, y[i], "-o", label="z=" + str(z[z_nodes[i]]))
        ax.legend()

    model_plot_yCO2theta_RKH(blk)

    plt.show()


def get_init_factors(blk):
    d1 = {
        "HT_gs": value(blk.R_HT_gs),
        "HT_ghx": value(blk.R_HT_ghx),
        "delH": value(blk.R_delH),
        "dP": value(blk.R_dP),
        "MT_gas": value(blk.R_MT_gas),
        "MT_solid": value(blk.R_MT_solid),
        "MT_coeff": value(blk.R_MT_coeff),
    }

    for key, v in d1.items():
        print(f"{key}: {v}")


def evaluate_MB_error(blk):
    for k in blk.component_list:
        # print error for each component formatted in scientific notation
        print(f"{k} error = {blk.MB_error[k]():.3} %")


def save_model(blk):
    to_json(blk, fname="RPB_model_051823.json.gz", gz=True, human_read=False)


def load_model(blk):
    from_json(blk, fname="RPB_model_051723.json.gz", gz=True)


def fix_BCs(blk):
    # blk.F_in.fix()
    blk.P_in.fix(1.1)
    blk.Tg_in.fix()
    blk.y_in.fix()

    blk.P_out.fix(1.01325)


def initial_solve(blk):
    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()


def homotopy_solve1(blk):
    solver = SolverFactory("ipopt")
    solver.options = {
        "warm_start_init_point": "yes",
        "bound_push": 1e-22,
        "nlp_scaling_method": "user-scaling",
        "max_iter": 1000,
        # 'halt_on_ampl_error': 'yes',
    }

    hom_points = np.logspace(-3, -1, 10)
    j = 0
    for i in hom_points:
        j += 1
        print("point ", j)
        print("init. point =", i)
        blk.R_HT_gs = i
        blk.R_HT_ghx = i
        blk.R_delH = i
        blk.R_MT_coeff = i
        solver.solve(blk, tee=True).write()


def custom_initialization_routine_1(blk):
    print("\nFixing Inlets to make square problem")
    fix_BCs(blk)

    print("degrees of freedom =", degrees_of_freedom(blk))

    print("\nInitial Solve")
    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()

    # homotopy solve 1
    print("\nHomotopy solve 1")
    solver.options = {
        "warm_start_init_point": "yes",
        "bound_push": 1e-22,
        "nlp_scaling_method": "user-scaling",
        "max_iter": 1000,
        # 'halt_on_ampl_error': 'yes',
    }

    hom_points = np.logspace(-3, -1, 5)
    j = 0
    for i in hom_points:
        j += 1
        print("point ", j)
        print("init. point =", i)
        blk.R_HT_gs = i
        blk.R_HT_ghx = i
        blk.R_delH = i
        blk.R_MT_coeff = i
        solver.solve(blk, tee=True).write()

    print("\nHomotopy solve 2")
    hom_points = np.linspace(0.1, 1, 5)
    for i in hom_points:
        print("init. point =", i)
        blk.R_HT_gs = i
        blk.R_HT_ghx = i
        blk.R_delH = i
        blk.R_MT_coeff = i
        solver.solve(blk, tee=True).write()

    print("\nHomotopy solve 3")
    hom_points = np.logspace(-3, -1, 5)
    for i in hom_points:
        print("init. point =", i)
        blk.R_MT_gas = i
        blk.R_MT_solid = i
        solver.solve(blk, tee=True).write()

    print("\nHomotopy solve 4")
    hom_points = np.linspace(0.1, 1, 10)
    for i in hom_points:
        print("init. point =", i)
        blk.R_MT_gas = i
        blk.R_MT_solid = i
        solver.solve(blk, tee=True).write()


def homotopy_init_routine(blk):
    print("\nFixing boundaries to make square problem")

    blk.P_in.fix(1.1)
    blk.Tg_in.fix()
    blk.y_in.fix()
    blk.P_out.fix(1.01325)

    print(f"DOF = {degrees_of_freedom(blk)}")

    variables_list = [
        blk.R_HT_gs,
        blk.R_HT_ghx,
        blk.R_delH,
        blk.R_MT_coeff,
        blk.R_MT_gas,
        blk.R_MT_solid,
    ]

    targets_list = [
        1,
        1,
        1,
        1,
        1,
        1,
    ]

    # homotopy solver
    homotopy(
        blk,
        variables_list,
        targets_list,
        max_solver_iterations=100,
        max_solver_time=60,
        min_step=0.01,
        iter_target=8,
    )


# Degeneracy Hunter
def degen_hunter(blk):
    dh = DegeneracyHunter(blk, solver=SolverFactory("cbc"))

    # various functions
    dh.check_residuals(tol=1e-6)
    dh.check_variable_bounds(tol=1e-6)
    # n_deficient = dh.check_rank_equality_constraints()


# check scaling
def check_scaling(blk):
    jac, nlp = iscale.get_jacobian(blk)

    # print("Extreme Jacobian entries:")
    with open("extreme_jacobian_entries.txt", "w") as f:
        for i in iscale.extreme_jacobian_entries(
            jac=jac, nlp=nlp, small=5e-3, large=1e3
        ):
            print(f"    {i[0]:.2e}, [{i[1]}, {i[2]}]", file=f)

    # print("Extreme Jacobian Columns:")
    with open("extreme_jacobian_columns.txt", "w") as f:
        for i in iscale.extreme_jacobian_columns(
            jac=jac, nlp=nlp, small=0.1, large=1e3
        ):
            print(f"    {i[0]:.2e}, [{i[1]}]", file=f)

    # print("Extreme Jacobian Rows:")
    with open("extreme_jacobian_rows.txt", "w") as f:
        for i in iscale.extreme_jacobian_rows(jac=jac, nlp=nlp, small=0.1, large=1e3):
            print(f"    {i[0]:.2e}, [{i[1]}]", file=f)

    with open("badly_scaled_vars.txt", "w") as f:
        for v, sv in iscale.badly_scaled_var_generator(
            blk, large=1e2, small=1e-1, zero=1e-12
        ):
            print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}", file=f)

    print(f"Jacobian Condition Number: {iscale.jacobian_cond(jac=jac):.2e}")


# Doug script
def scaling_script(blk):
    # import numpy as np
    from scipy.linalg import svd

    # import pyomo.environ as pyo
    # import idaes.core.util.scaling as iscale

    jac, nlp = iscale.get_jacobian(blk)

    variables = nlp.get_pyomo_variables()
    constraints = nlp.get_pyomo_equality_constraints()
    print("Badly scaled variables:")
    for i in iscale.extreme_jacobian_columns(jac=jac, nlp=nlp, large=1e3, small=5e-3):
        print(f"    {i[0]:.2e}, [{i[1]}]")
    print("\n\n" + "Badly scaled constraints:")
    for i in iscale.extreme_jacobian_rows(jac=jac, nlp=nlp, large=1e3, small=5e-3):
        print(f"    {i[0]:.2e}, [{i[1]}]")
    # print(f"Jacobian Condition Number: {iscale.jacobian_cond(jac=jac):.2e}")
    # if not hasattr(m.fs, "obj"):
    #     m.fs.obj = pyo.Objective(expr=0)
    n_sv = 10
    u, s, vT = svd(jac.todense(), full_matrices=False)

    print("\n" + f"Spectral condition number: {s[0]/s[-1]:.3e}")
    # Reorder singular values and vectors so that the singular
    # values are from least to greatest
    u = np.flip(u[:, -n_sv:], axis=1)
    s = np.flip(s[-n_sv:], axis=0)
    vT = np.flip(vT[-n_sv:, :], axis=0)
    v = vT.transpose()
    print("\n" + f"Smallest singular value: {s[0]}")
    print("\n" + "Variables in smallest singular vector:")
    for i in np.where(abs(v[:, 0]) > 0.1)[0]:
        print(str(i) + ": " + variables[i].name)
    print("\n" + "Constraints in smallest singular vector:")
    for i in np.where(abs(u[:, 0]) > 0.1)[0]:
        print(str(i) + ": " + constraints[i].name)

    return jac, variables, constraints


def single_section_init(blk):
    init_obj = BlockTriangularizationInitializer()
    init_obj.config.block_solver_call_options = {"tee": True}

    blk.P_in.fix(1.1)
    blk.Tg_in.fix()
    blk.y_in.fix()
    blk.P_out.fix(1.01325)

    blk.R_MT_solid = 1
    blk.R_MT_gas = 1

    init_obj.initialization_routine(blk)

    blk.R_MT_coeff = 1
    blk.R_HT_ghx = 1
    blk.R_HT_gs = 1
    blk.R_delH = 1

    init_obj.initialization_routine(blk)

    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()


def full_model_creation(lean_temp_connection=True):
    RPB = ConcreteModel()

    RPB.ads = RPB_model(mode="adsorption")
    RPB.des = RPB_model(mode="desorption")

    # fix BCs
    # RPB.ads.P_in.fix(1.1)
    RPB.ads.P_in.fix(1.025649)
    RPB.ads.Tg_in.fix()
    RPB.ads.y_in.fix()
    RPB.ads.P_out.fix(1.01325)

    RPB.des.P_in.fix(1.1)
    RPB.des.Tg_in.fix()
    RPB.des.y_in.fix()
    RPB.des.P_out.fix(1.01325)

    # connect rich stream
    # unfix inlet loading and temperature to the desorption section. (No mass transfer at boundaries so z=0 and z=1 need to remain fixed.)
    for z in RPB.des.z:
        if z != 0 and z != 1:
            RPB.des.qCO2_in[z].unfix()
            RPB.des.Ts_in[z].unfix()

    # add equality constraint equating inlet desorption loading to outlet adsorption loading. Same for temperature.
    @RPB.Constraint(RPB.des.z)
    def rich_loading_constraint(RPB, z):
        if z == 0 or z == 1:
            return Constraint.Skip
        else:
            return RPB.des.qCO2_in[z] == RPB.ads.qCO2[z, 1]

    @RPB.Constraint(RPB.des.z)
    def rich_temp_constraint(RPB, z):
        if z == 0 or z == 1:
            return Constraint.Skip
        else:
            return 1e-2 * RPB.des.Ts_in[z] == 1e-2 * RPB.ads.Ts[z, 1]

    # connect lean stream
    # unfix inlet loading to the adsorption section
    for z in RPB.ads.z:
        if z != 0 and z != 1:
            RPB.ads.qCO2_in[z].unfix()
            if lean_temp_connection:
                RPB.ads.Ts_in[z].unfix()

    # add equality constraint equating inlet adsorption loading to outlet desorption loading
    @RPB.Constraint(RPB.ads.z)
    def lean_loading_constraint(RPB, z):
        if z == 0 or z == 1:
            return Constraint.Skip
        else:
            return RPB.des.qCO2[z, 1] == RPB.ads.qCO2_in[z]

    if lean_temp_connection:

        @RPB.Constraint(RPB.ads.z)
        def lean_temp_constraint(RPB, z):
            if z == 0 or z == 1:
                return Constraint.Skip
            else:
                return 1e-2 * RPB.des.Ts[z, 1] == 1e-2 * RPB.ads.Ts_in[z]

    # add constraints so that the length, diameter, and rotational speed are the same for both sides
    RPB.des.L.unfix()  # unfix des side vars
    RPB.des.D.unfix()
    RPB.des.w_rpm.unfix()

    @RPB.Constraint(doc="Length equality constraint")
    def length_constraint(RPB):
        return RPB.ads.L == RPB.des.L

    @RPB.Constraint(doc="Diameter equality constraint")
    def diameter_constraint(RPB):
        return RPB.ads.D == RPB.des.D

    @RPB.Constraint(doc="Rotational speed equality constraint")
    def speed_constraint(RPB):
        return RPB.ads.w_rpm == RPB.des.w_rpm

    # add constraint so that the fraction of each section adds to 1
    RPB.des.theta.unfix()  # unfix des side var

    @RPB.Constraint(doc="Theta summation constraint")
    def theta_constraint(RPB):
        return RPB.ads.theta + RPB.des.theta == 1

    return RPB


def init_routine_1(blk):
    # create Block init object
    init_obj = BlockTriangularizationInitializer()

    init_obj.config.block_solver_call_options = {"tee": True}
    init_obj.config.block_solver_options = {
        "halt_on_ampl_error": "yes",
        "max_iter": 500,
    }

    # turn on solids mass transfer (with the loadings connected at the rich and lean ends, solids mass transfer has to be turned on or no solution exists)
    blk.ads.R_MT_solid = 1
    blk.des.R_MT_solid = 1

    # run initialization routine

    init_obj.initialization_routine(blk)

    blk.ads.R_MT_gas = 1
    blk.des.R_MT_gas = 1
    blk.ads.R_MT_coeff = 1
    blk.des.R_MT_coeff = 1
    blk.ads.R_HT_ghx = 1
    blk.des.R_HT_ghx = 1
    blk.ads.R_HT_gs = 1
    blk.des.R_HT_gs = 1
    blk.ads.R_delH = 1
    blk.des.R_delH = 1
    init_obj.initialization_routine(blk)

    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()


def fix_capture_and_solve(blk, capture=0.9):
    fix_capture(blk, capture=capture)

    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        # "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()


def fix_capture(blk, capture=0.9):
    blk.ads.P_in.unfix()
    blk.ads.CO2_capture.fix(capture)


def report(blk):
    items = [
        blk.ads.L,
        blk.ads.D,
        blk.ads.w_rpm,
        blk.ads.theta,
        blk.des.theta,
        blk.ads.P_in,
        blk.ads.P_out,
        blk.ads.F_in,
        blk.ads.Tg_in,
        blk.ads.Tx,
        blk.des.P_in,
        blk.des.P_out,
        blk.des.F_in,
        blk.des.Tg_in,
        blk.des.Tx,
        blk.ads.CO2_capture,
    ]

    names = []
    values = []
    fixed = []
    docs = []
    for item in items:
        names.append(item.to_string())
        values.append(item())
        fixed.append(item.fixed)
        docs.append(item.doc)

    report_df = pd.DataFrame(
        data={"Value": values, "Doc": docs, "Fixed": fixed}, index=names
    )

    indexed_items = [
        blk.ads.y_in,
        blk.ads.y_out,
    ]

    names = []
    values = []
    docs = []
    fixed = []
    for item in indexed_items:
        names += [item[k].to_string() for k in item.keys()]
        values += [item[k]() for k in item.keys()]
        docs += [item.doc for k in item.keys()]
        fixed += [item[k].fixed for k in item.keys()]

    report_indexed_df = pd.DataFrame(
        data={"Value": values, "Doc": docs, "Fixed": fixed}, index=names
    )

    report_df = pd.concat([report_df, report_indexed_df])

    return report_df


def full_contactor_plotting(blk, save_option=False):
    z = list(blk.ads.z)
    theta = list(blk.ads.o)

    theta_total = list(blk.ads.o)
    for o in list(blk.ads.o):
        if o != 0:
            theta_total.append(1 + o)

    theta_total_norm = [k / max(theta_total) for k in theta_total]

    z_query = [0.05, 0.15, 0.5, 0.75, 0.9]
    z_nodes = [blk.ads.z.find_nearest_index(z) for z in z_query]

    theta_query = [0.05, 0.15, 0.5, 0.75, 0.9]
    theta_nodes = [blk.ads.o.find_nearest_index(o) for o in theta_query]

    # Solids Loading
    qCO2_ads = [[], [], [], [], []]
    qCO2_des = [[], [], [], [], []]
    qCO2_total = [[], [], [], [], []]
    k = 0
    for j in z_nodes:
        for i in theta:
            qCO2_ads[k].append(blk.ads.qCO2[z[j], i]())
            qCO2_des[k].append(blk.des.qCO2[z[j], i]())
        k += 1

    for k in range(len(z_nodes)):
        qCO2_total[k] = qCO2_ads[k] + qCO2_des[k][1:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Theta distance (radians)", fontsize=16)
    ax.set_ylabel("CO$_{2}$ Loading [mol/kg]", fontsize=16)
    # ax.set_title('Adsorption CO$_{2}$ Loading')
    for i in range(len(z_nodes)):
        ax.plot(
            theta_total_norm,
            qCO2_total[i],
            "-o",
            label="z=" + str(round(z[z_nodes[i]], 3)),
        )
    ax.axvline(x=blk.ads.theta(), color="k", linestyle="--")
    # ymin, ymax = ax.get_ylim()
    # ax.text(
    #     0.1,
    #     0.5 * (ymax - ymin) + ymin,
    #     "Adsorption Section",
    #     bbox=dict(facecolor="white", alpha=0.5),
    # )
    # ax.text(
    #     0.6,
    #     0.5 * (ymax - ymin) + ymin,
    #     "Desorption Section",
    #     bbox=dict(facecolor="white", alpha=0.5),
    # )
    ax.legend()

    if save_option:
        fig.savefig("CO2_loading.png", dpi=300)

    # Solids temperature
    Ts_ads = [[], [], [], [], []]
    Ts_des = [[], [], [], [], []]
    Ts_total = [[], [], [], [], []]
    k = 0
    for j in z_nodes:
        for i in theta:
            Ts_ads[k].append(blk.ads.Ts[z[j], i]())
            Ts_des[k].append(blk.des.Ts[z[j], i]())
        k += 1

    for k in range(len(z_nodes)):
        Ts_total[k] = Ts_ads[k] + Ts_des[k][1:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Theta distance (radians)", fontsize=16)
    ax.set_ylabel("Solids Temperature [K]", fontsize=16)
    # ax.set_title('Adsorption CO$_{2}$ Loading')
    for i in range(len(z_nodes)):
        ax.plot(
            theta_total_norm,
            Ts_total[i],
            "-o",
            label="z=" + str(round(z[z_nodes[i]], 3)),
        )
    ax.axvline(x=blk.ads.theta(), color="k", linestyle="--")
    # ymin, ymax = ax.get_ylim()
    # ax.text(
    #     0.1,
    #     0.5 * (ymax - ymin) + ymin,
    #     "Adsorption Section",
    #     bbox=dict(facecolor="white", alpha=0.5),
    # )
    # ax.text(
    #     0.6,
    #     0.5 * (ymax - ymin) + ymin,
    #     "Desorption Section",
    #     bbox=dict(facecolor="white", alpha=0.5),
    # )
    ax.legend()

    if save_option:
        fig.savefig("solid temp.png", dpi=300)

    # Adsorber Gas phase CO2 mole fraction
    y_CO2 = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            y_CO2[k].append(blk.ads.y["CO2", i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas phase CO$_{2}$ mole fraction, Adsorber", fontsize=12)
    # ax.set_ylim([0, 0.05])
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(
            z, y_CO2[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3))
        )
    ax.legend()

    if save_option:
        fig.savefig("CO2_molefraction_ads.png", dpi=300)

    # Desorber Gas phase CO2 mole fraction
    y_CO2 = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            y_CO2[k].append(blk.des.y["CO2", i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas phase CO$_{2}$ mole fraction, Desorber", fontsize=12)
    # ax.set_ylim([0, 0.05])
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(
            z, y_CO2[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3))
        )
    ax.legend()

    if save_option:
        fig.savefig("CO2_molefraction_des.png", dpi=300)

    # Adsorber Gas Temperature
    Tg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Tg[k].append(blk.ads.Tg[i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas Temperature, Adsorber [K]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, Tg[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.legend()

    if save_option:
        fig.savefig("GasTemp_ads.png", dpi=300)

    # Desorber Gas Temperature
    Tg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Tg[k].append(blk.des.Tg[i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas Temperature, Desorber [K]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, Tg[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.legend()

    if save_option:
        fig.savefig("GasTemp_des.png", dpi=300)

    # Adsorber Gas Pressure
    Pg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Pg[k].append(blk.ads.P[i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas Pressure, Adsorber [bar]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, Pg[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.legend()

    if save_option:
        fig.savefig("GasPress_ads.png", dpi=300)

    # Desorber Gas Pressure
    Pg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Pg[k].append(blk.des.P[i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas Pressure, Desorber [bar]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, Pg[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.legend()

    if save_option:
        fig.savefig("GasPress_des.png", dpi=300)

    # Adsorber Gas Velocity
    vel = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            vel[k].append(blk.ads.vel[i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas velocity, Adsorber [m/s]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, vel[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.legend()

    if save_option:
        fig.savefig("GasVel_ads.png", dpi=300)

    # Desorber Gas Velocity
    vel = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            vel[k].append(blk.des.vel[i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas velocity, Desorber [m/s]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, vel[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.legend()

    if save_option:
        fig.savefig("GasVel_des.png", dpi=300)

    plt.show()
