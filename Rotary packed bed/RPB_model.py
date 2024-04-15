#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

"""
Rotary Packed Bed Model
"""

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Constraint,
    units,
    exp,
    log,
    TransformationFactory,
    SolverFactory,
    value,
    Set,
    Objective,
    Block,
    PositiveReals,
    NonNegativeReals,
    Reals,
)
from pyomo.network import Port
from pyomo.dae import ContinuousSet, DerivativeVar
from idaes import *
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import to_json, from_json, StoreSpec
import idaes.core.util.scaling as iscale
from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.models.unit_models import SkeletonUnitModel

from idaes.core import declare_process_block_class, UnitModelBlockData, useDefault

import finitevolume
from idaes.core.solvers.homotopy import homotopy
from idaes.core.initialization.block_triangularization import (
    BlockTriangularizationInitializer,
)

from pyomo.common.config import ConfigBlock, ConfigValue, In

from idaes.core.util.math import smooth_max
from idaes.core.util.constants import Constants as const

from idaes.models_extra.power_generation.properties import FlueGasParameterBlock

import idaes.logger as idaeslog

__author__ = "Ryan Hughes"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("RPB")
class RPBData(UnitModelBlockData):
    """
    Standard Rotary Packed Bed Unit Model Class
    """

    CONFIG = UnitModelBlockData.CONFIG()

    CONFIG.declare(
        "z_init_points",
        ConfigValue(
            default=tuple(np.geomspace(0.01, 0.5, 9)[:-1])
            + tuple((1 - np.geomspace(0.01, 0.5, 9))[::-1]),
            domain=tuple,
            description="initial axial nodes",
        ),
    )

    CONFIG.declare(
        "z_disc_method",
        ConfigValue(
            default="Finite Difference",
            domain=In(["Finite Difference", "Collocation"]),
            description="Axial discretization method",
        ),
    )

    CONFIG.declare(
        "z_nfe",
        ConfigValue(
            default=20,
            domain=int,
            description="Number of finite elements for axial direction",
        ),
    )

    CONFIG.declare(
        "z_collocation_points",
        ConfigValue(
            default=2,
            domain=int,
            description="Number of collocation points for axial direction",
        ),
    )

    def build(self):
        """
        General build method for RPB

        Inheriting models should call `super().build`.

        Args:
            None

        Returns:
            None

        """

        # call UnitModel.build to build default attributes
        super().build()

        # Add general parameters
        self._add_general_parameters()

        # Add sorbent parameters
        self._add_sorbent_parameters()

        # Add expressions for constants
        @self.Expression(doc="gas constant [kJ/mol/K]")
        def R(b):
            return units.convert(
                const.gas_constant, to_units=units.kJ / units.mol / units.K
            )

        @self.Expression(doc="gas constant [m^3*bar/K/mol]")
        def Rg(b):
            return units.convert(
                const.gas_constant,
                to_units=units.m**3 * units.bar / units.K / units.mol,
            )

        # add design and operating variables
        self.D = Var(
            initialize=(10), domain=PositiveReals, units=units.m, doc="Bed diameter [m]"
        )
        self.L = Var(
            initialize=(3),
            domain=PositiveReals,
            bounds=(0.1, 10.001),
            units=units.m,
            doc="Bed Length [m]",
        )

        self.w_rpm = Var(
            self.flowsheet().time,
            initialize=(1),
            domain=PositiveReals,
            bounds=(0.00001, 2),
            units=units.revolutions / units.min,
            doc="bed rotational speed [revolutions/min]",
        )

        @self.Expression(self.flowsheet().time, doc="bed rotational speed [radians/s]")
        def w(b, t):
            return (
                b.w_rpm[t]
                * (2 * const.pi * units.radians / units.revolutions)
                / (60 * units.sec / units.min)
            )

    def _add_general_parameters(self):
        """
        Method to add general parameters to the RPB model. This includes ...
        """

        self.component_list = Set(
            initialize=["N2", "CO2", "H2O"], doc="List of components"
        )

        self.ads_components = Set(
            initialize=["CO2"],
            within=self.component_list,
            doc="list of adsorbing components",
        )

        self.MW = Param(
            self.component_list,
            initialize=({"CO2": 44.01e-3, "N2": 28.0134e-3, "H2O": 18.01528e-3}),
            units=units.kg / units.mol,
            doc="component molecular weight [kg/mol]",
        )

        self.mu = Param(
            self.component_list,
            initialize=({"CO2": 1.87e-5, "N2": 2.3e-5, "H2O": 1.26e-5}),
            units=units.Pa * units.s,
            doc="pure component gas phase viscosity [Pa*s]",
        )

        self.k = Param(
            self.component_list,
            initialize=({"CO2": 0.020e-3, "N2": 0.030e-3, "H2O": 0.025e-3}),
            units=units.kJ / units.s / units.m / units.K,
            doc="pure component gas phase thermal conductivity [kW/m/K, kJ/s/m/K]",
        )

        self.DmCO2 = Param(
            initialize=(5.3e-5),
            units=units.m**2 / units.s,
            doc="gas phase CO2 diffusivity [m^2/s]",
        )

    def _add_sorbent_parameters(self):
        """
        Method to add sorbent parameters (and accompanying expressions) to the RPB model. This includes ...
        """

        # general parameters
        self.eb = Param(initialize=(0.68), doc="bed voidage")
        self.ep = Param(initialize=(0.68), doc="particle porosity")
        self.dp = Param(
            initialize=(0.000525), units=units.m, doc="particle diameter [m]"
        )
        self.Cp_sol = Param(
            initialize=(1.457),
            units=units.kJ / units.kg / units.K,
            doc="solid heat capacity [kJ/kg/K]",
        )
        self.rho_sol = Param(
            initialize=(1144),
            units=units.kg / units.m**3,
            mutable=True,
            doc="solid particle densitry [kg/m^3]",
        )

        @self.Expression(doc="particle radius [m]")
        def rp(b):
            return b.dp / 2

        @self.Expression(
            doc="specific particle area for mass transfer, bed voidage"
            "included [m^2/m^3 bed]"
        )
        def a_s(b):
            return 6 / b.dp * (1 - b.eb)

        # Isotherm Parameters
        self.q_inf_1 = Param(
            initialize=2.87e-02, units=units.mol / units.kg, doc="isotherm parameter"
        )
        self.q_inf_2 = Param(
            initialize=1.95, units=units.mol / units.kg, doc="isotherm parameter"
        )
        self.q_inf_3 = Param(
            initialize=3.45, units=units.mol / units.kg, doc="isotherm parameter"
        )

        self.d_inf_1 = Param(
            initialize=1670.31, units=units.bar**-1, doc="isotherm parameter"
        )
        self.d_inf_2 = Param(
            initialize=789.01, units=units.bar**-1, doc="isotherm parameter"
        )
        self.d_inf_3 = Param(
            initialize=10990.67, units=units.bar**-1, doc="isotherm parameter"
        )
        self.d_inf_4 = Param(
            initialize=0.28,
            units=units.mol / units.kg / units.bar,
            doc="isotherm parameter",
        )

        self.E_1 = Param(
            initialize=-76.15, units=units.kJ / units.mol, doc="isotherm parameter"
        )
        self.E_2 = Param(
            initialize=-77.44, units=units.kJ / units.mol, doc="isotherm parameter"
        )
        self.E_3 = Param(
            initialize=-194.48, units=units.kJ / units.mol, doc="isotherm parameter"
        )
        self.E_4 = Param(
            initialize=-6.76, units=units.kJ / units.mol, doc="isotherm parameter"
        )

        self.X_11 = Param(initialize=4.20e-2, doc="isotherm parameter")
        self.X_21 = Param(initialize=2.97, units=units.K, doc="isotherm parameter")
        self.X_12 = Param(initialize=7.74e-2, doc="isotherm parameter")
        self.X_22 = Param(initialize=1.66, units=units.K, doc="isotherm parameter")

        self.P_step_01 = Param(
            initialize=1.85e-03, units=units.bar, doc="isotherm parameter"
        )
        self.P_step_02 = Param(
            initialize=1.78e-02, units=units.bar, doc="isotherm parameter"
        )

        @self.Expression()
        def ln_P0_1(b):
            return log(b.P_step_01 / units.bar)

        @self.Expression()
        def ln_P0_2(b):
            return log(b.P_step_02 / units.bar)

        self.H_step_1 = Param(
            initialize=-99.64, units=units.kJ / units.mol, doc="isotherm parameter"
        )
        self.H_step_2 = Param(
            initialize=-78.19, units=units.kJ / units.mol, doc="isotherm parameter"
        )

        self.gamma_1 = Param(initialize=894.67, doc="isotherm parameter")
        self.gamma_2 = Param(initialize=95.22, doc="isotherm parameter")

        self.T0 = Param(initialize=363.15, units=units.K, doc="isotherm parameter")

        # Mass transfer parameters
        self.C1 = Param(
            initialize=(2.562434e-12),
            units=units.m**2 / units.K**0.5 / units.s,
            doc="lumped MT parameter [m^2/K^0.5/s]",
        )

        # heat of adsorption parameters
        self.delH_a1 = Param(
            initialize=21.68,
            units=units.kg / units.mol,
            doc="heat of adsorption parameter",
        )
        self.delH_a2 = Param(
            initialize=29.10,
            units=units.kg / units.mol,
            doc="heat of adsorption parameter",
        )
        self.delH_b1 = Param(
            initialize=1.59,
            units=units.mol / units.kg,
            doc="heat of adsorption parameter",
        )
        self.delH_b2 = Param(
            initialize=3.39,
            units=units.mol / units.kg,
            doc="heat of adsorption parameter",
        )
        self.delH_1 = Param(
            initialize=98.76,
            units=units.kJ / units.mol,
            doc="heat of adsorption parameter",
        )
        self.delH_2 = Param(
            initialize=77.11,
            units=units.kJ / units.mol,
            doc="heat of adsorption parameter",
        )
        self.delH_3 = Param(
            initialize=21.25,
            units=units.kJ / units.mol,
            doc="heat of adsorption parameter",
        )

    def _add_section(
        self, name, gas_flow_direction="forward", initial_guesses="Adsorption"
    ):
        """
        Method to add a single section to the RPB model. This includes ...

        Args:
            name : str : name of the section
            gas_flow_direction : str : direction of gas flow in the section
            initial_guesses : str : initial guesses keyword for variables in the section

        Returns:
            None

        """

        setattr(self, name, SkeletonUnitModel())
        blk = getattr(self, name)

        blk.CONFIG.declare(
            "gas_flow_direction",
            ConfigValue(
                default=gas_flow_direction,
                domain=In(["forward", "reverse"]),
                description="gas flow direction, used for simulation of counter-current configuration. Forward flows from 0 to 1, reverse flows from 1 to 0",
            ),
        )

        blk.CONFIG.declare(
            "o_init_points",
            ConfigValue(
                default=tuple(np.geomspace(0.005, 0.1, 8))
                + tuple(np.linspace(0.1, 0.995, 10)[1:]),
                domain=tuple,
                description="initial o nodes",
            ),
        )

        blk.CONFIG.declare(
            "o_nfe",
            ConfigValue(
                default=20, domain=int, description="Number of o finite elements"
            ),
        )

        blk.CONFIG.declare(
            "o_collocation_points",
            ConfigValue(
                default=2, domain=int, description="Number of o collocation points"
            ),
        )

        blk.CONFIG.declare(
            "o_disc_method",
            ConfigValue(
                default="Finite Difference",
                domain=In(["Finite Difference", "Collocation"]),
                description="o discretization method",
            ),
        )

        blk.z = ContinuousSet(
            doc="axial dimension [dimensionless]",
            bounds=(0, 1),
            initialize=self.CONFIG.z_init_points,
        )

        blk.o = ContinuousSet(
            doc="adsorption theta nodes [dimensionless]",
            bounds=(0, 1),
            initialize=blk.CONFIG.o_init_points,
        )

        # add section parameters
        if initial_guesses == "adsorption":
            theta_0 = 0.75
        elif initial_guesses == "desorption":
            theta_0 = 0.25
        else:
            theta_0 = 0.5

        blk.theta = Var(
            initialize=(theta_0),
            domain=PositiveReals,
            bounds=(0.01, 0.99),
            doc="Fraction of bed occupied by the section[-]",
        )

        # embedded heat exchanger and area calculations
        blk.Hx_frac = Param(
            initialize=(1 / 3),  # current assumption, HE takes up 1/3 of bed
            mutable=True,
            doc="fraction of total reactor volume occupied by the embedded"
            "heat exchanger",
        )

        blk.a_sp = Param(
            initialize=(50),
            units=units.m**2 / units.m**3,
            doc="specific surface area for heat transfer [m^2/m^3]",
        )

        @blk.Expression(doc="cross sectional area, total area*theta [m^2]")
        def A_c(b):
            return const.pi * self.D**2 / 4 * b.theta

        @blk.Expression(doc="cross sectional area for flow [m^2]")
        def A_b(b):
            return (1 - b.Hx_frac) * b.A_c  # current assumption, HE takes up 1/3 of bed

        @blk.Expression(doc="cross sectional area of the heat exchanger [m^2]")
        def Ahx(b):
            return b.Hx_frac * b.A_c

        @blk.Expression(doc="specific heat transfer area [m^2/m^3]")
        def a_ht(b):  # current assumption, can update this later
            return b.a_sp

        # ============================ Gas Inlet =======================================
        blk.F_in = Var(
            self.flowsheet().time,
            initialize=400,
            domain=PositiveReals,
            doc="Inlet adsorber gas flow [mol/s]",
            bounds=(0, None),
            units=units.mol / units.s,
        )

        blk.P_in = Var(
            self.flowsheet().time,
            initialize=1.1,
            domain=PositiveReals,
            bounds=(1, 1.5),
            units=units.bar,
            doc="Inlet flue gas pressure [bar]",
        )

        if initial_guesses == "adsorption":
            Tg_in = 90 + 273
            y_in = {(0, "CO2"): 0.04, (0, "N2"): 0.87, (0, "H2O"): 0.09}
        elif initial_guesses == "desorption":
            Tg_in = 120 + 273
            y_in = {(0, "CO2"): 1e-5, (0, "N2"): 1e-3, (0, "H2O"): (1 - 1e-5 - 1e-3)}
        else:
            Tg_in = 90 + 273
            y_in = {(0, "CO2"): 0.04, (0, "N2"): 0.87, (0, "H2O"): 0.09}

        blk.Tg_in = Var(
            self.flowsheet().time,
            initialize=Tg_in,
            domain=PositiveReals,
            units=units.K,
            doc="Inlet flue gas temperature [K]",
        )

        blk.y_in = Var(
            self.flowsheet().time,
            self.component_list,
            initialize=y_in,
            domain=PositiveReals,
            doc="inlet mole fraction",
        )

        # add inlet port
        p = Port(noruleinit=True, doc="Inlet gas port for RPB")
        setattr(blk, "inlet", p)
        inlet_dict = {
            "F_in": blk.F_in,
            "P_in": blk.P_in,
            "Tg_in": blk.Tg_in,
            "y_in": blk.y_in,
        }
        for k in inlet_dict.keys():
            p.add(inlet_dict[k], name=k)

        # Inlet values for initialization
        @blk.Expression(self.flowsheet().time, doc="inlet total conc. [mol/m^3]")
        def C_tot_in(b, t):
            return b.P_in[t] / b.Tg_in[t] / self.Rg

        @blk.Expression(
            self.flowsheet().time,
            self.component_list,
            doc="inlet concentrations [mol/m^3]",
        )
        def C_in(b, t, k):
            return b.y_in[t, k] * b.C_tot_in[t]

        @blk.Expression(
            self.flowsheet().time, doc="inlet gas velocity, adsorption [m/s]"
        )
        def vel0(b, t):
            return b.F_in[t] / b.C_tot_in[t] / b.A_b

        # =========================== Gas Outlet =======================================
        blk.P_out = Var(
            self.flowsheet().time,
            initialize=1.01325,
            domain=PositiveReals,
            bounds=(0.99, 1.2),
            units=units.bar,
            doc="Outlet adsorber pressure [bar]",
        )

        blk.F_out = Var(
            self.flowsheet().time,
            initialize=blk.F_in[0](),
            domain=PositiveReals,
            bounds=(0, None),
            units=units.mol / units.s,
            doc="Total gas outlet flow [mol/s]",
        )

        blk.y_out = Var(
            self.flowsheet().time,
            self.component_list,
            domain=PositiveReals,
            bounds=(0, 1),
            initialize=y_in,
            doc="outlet mole fraction",
        )

        blk.Tg_out = Var(
            self.flowsheet().time,
            initialize=100 + 273.15,
            domain=PositiveReals,
            bounds=(25 + 273.15, 180 + 273.15),
            units=units.K,
            doc="outlet gas temperature [K]",
        )

        # add outlet port
        p = Port(noruleinit=True, doc="Outlet gas port for RPB")
        setattr(blk, "outlet", p)
        outlet_dict = {
            "F_out": blk.F_out,
            "P_out": blk.P_out,
            "Tg_out": blk.Tg_out,
            "y_out": blk.y_out,
        }
        for k in outlet_dict.keys():
            p.add(outlet_dict[k], name=k)

        # ======================== Heat exchanger ======================================
        blk.hgx = Param(
            initialize=(25 * 1e-3),  # assumed value
            units=units.kW / units.m**2 / units.K,
            doc="heat exchanger heat transfer coeff. kW/m^2/K",
        )

        if initial_guesses == "adsorption":
            Tx = 90 + 273
        elif initial_guesses == "desorption":
            Tx = 120 + 273
        else:
            Tx = 90 + 273

        blk.Tx = Var(
            self.flowsheet().time,
            initialize=Tx,
            domain=PositiveReals,
            units=units.K,
            doc="heat exchange fluid temperature, constant [K]",
        )
        # ==============================================================================

        # Variable declaration =========================================================
        # ============================== Gas Phase =====================================
        blk.Cs_r = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=blk.C_in[0, "CO2"](),
            domain=NonNegativeReals,
            bounds=(0, 100),
            units=units.mol / units.m**3,
            doc="particle surface concentration of CO2 [mol/m^3]",
        )

        blk.y = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            bounds=(0, 1),
            initialize=0.1,
            domain=NonNegativeReals,
            doc="gas phase mole fraction",
        )

        blk.C_tot = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=blk.C_tot_in[0](),
            bounds=(0, 100),
            domain=NonNegativeReals,
            doc="Total conc., [mol/m^3] (ideal gas law)",
            units=units.mol / units.m**3,
        )

        blk.Tg = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=blk.Tg_in[0](),
            domain=PositiveReals,
            bounds=(25 + 273.15, 180 + 273.15),
            doc="Gas phase temperature [K]",
            units=units.K,
        )

        blk.heat_flux = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=0,
            units=units.kJ / units.m**2 / units.s,
            doc="heat flux [kW/m^2 or kJ/s/m^2]",
        )

        blk.dheat_fluxdz = DerivativeVar(
            blk.heat_flux,
            wrt=blk.z,
            units=units.kW / units.m**2,
            doc="axial derivative of heat flux [kW/m^2/dimensionless bed length]",
        )

        blk.vel = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=blk.vel0[0](),
            bounds=(0, 5),
            domain=NonNegativeReals,
            units=units.m / units.s,
            doc="superficial gas velocity [m/s], adsorption",
        )

        blk.P = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=blk.P_in[0].value,
            bounds=(0.99, 1.2),
            domain=PositiveReals,
            units=units.bar,
            doc="Gas Pressure [bar]",
        )

        blk.dPdz = DerivativeVar(
            blk.P,
            wrt=blk.z,
            bounds=(-1, 1),
            units=units.bar,
            doc="axial derivative of pressure [bar/dimensionless bed length]",
        )

        blk.Flux_kzo = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            bounds=(-1000, 1000),
            units=units.mol / units.m**2 / units.s,
            doc="Gas phse component flux [mol/m^2/s]",
        )

        blk.dFluxdz = DerivativeVar(
            blk.Flux_kzo,
            wrt=blk.z,
            units=units.mol / units.m**2 / units.s,
            doc="axial derivative of component flux [mol/m^2 bed/s/dimensionless bed length]",
        )

        # ========================= Solids =============================================
        if initial_guesses == "adsorption":
            qCO2_in_init = 1
            Ts_in_init = 100 + 273
        elif initial_guesses == "desorption":
            qCO2_in_init = 2.5
            Ts_in_init = 110 + 273
        else:
            qCO2_in_init = 1
            Ts_in_init = 100 + 273

        blk.qCO2 = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=qCO2_in_init,
            domain=NonNegativeReals,
            bounds=(0, 5),
            doc="CO2 loading [mol/kg]",
            units=units.mol / units.kg,
        )

        blk.dqCO2do = DerivativeVar(
            blk.qCO2,
            wrt=blk.o,
            units=units.mol / units.kg,
            doc="theta derivative of loading [mol/kg/dimensionless bed fraction]",
        )

        blk.Ts = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=Ts_in_init,
            domain=PositiveReals,
            bounds=(25 + 273.15, 180 + 273.15),
            doc="solid phase temperature [K], adsorption",
            units=units.K,
        )

        blk.dTsdo = DerivativeVar(
            blk.Ts,
            wrt=blk.o,
            units=units.K,
            doc="theta derivative of solid phase temp. [K/dimensionless bed fraction]",
        )
        # ==============================================================================

        # Initialization factors =======================================================
        blk.R_MT_gas = Var(
            initialize=1,
            doc="init. factor for mass transfer in gas phase MB (0=off, 1=on)",
        )
        blk.R_MT_gas.fix()

        blk.R_MT_solid = Var(
            initialize=1,
            doc="init. factor for mass transfer in solid phase MB (0=off, 1=on)",
        )
        blk.R_MT_solid.fix()

        blk.R_HT_gs = Var(
            initialize=1,
            doc="init. factor for gas-to-solid heat transfer (0=off, 1=on)",
        )
        blk.R_HT_gs.fix()

        blk.R_HT_ghx = Var(
            initialize=1, doc="init. factor for gas-to-HE heat transfer (0=off, 1=on)"
        )
        blk.R_HT_ghx.fix()

        blk.R_delH = Var(
            initialize=1, doc="init. factor for heat of adsorption (0=off, 1=on)"
        )
        blk.R_delH.fix()

        blk.R_dP = Var(initialize=1, doc="init. factor for pressure drop (0=off, 1=on)")
        blk.R_dP.fix()

        blk.R_MT_coeff = Var(
            initialize=1,
            doc="init. factor for the mass transfer coefficient (0=constant value, 1=model prediction)",
        )
        blk.R_MT_coeff.fix()
        # ==============================================================================

        # Gas phase equations ==========================================================
        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="total concentration equation (ideal gas law) [mol/m^3]",
        )
        def C_tot_eq(b, t, z, o):
            return b.C_tot[t, z, o] * self.Rg * b.Tg[t, z, o] == b.P[t, z, o]

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            doc="gas species concentration [mol/m^3]",
        )
        def C(b, t, z, o, k):
            return b.y[t, z, o, k] * b.C_tot[t, z, o]

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            wrt=blk.o,
            doc="Component flow integrated over theta, function of z [mol/s]",
        )
        def Flow_kz(b, t, z, o, k):
            return b.Flux_kzo[t, z, o, k] * b.A_b

        blk.Flow_z = Var(
            self.flowsheet().time,
            blk.z,
            initialize=blk.F_in[0](),
            bounds=(0, None),
            units=units.mol / units.s,
            doc="Total flow integrated over theta, function of z [mol/s]",
        )

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            doc="Total flow integrated over theta, function of z [mol/s]",
        )
        def Flow_z_eq(b, t, z):
            return b.Flow_z[t, z] == sum(
                b.Flow_kz[t, z, k] for k in self.component_list
            )

        blk.y_kz = Var(
            self.flowsheet().time,
            blk.z,
            self.component_list,
            initialize=0.1,
            bounds=(0, 1),
            doc="Component mole fraction integrated over theta, function of z [-]",
        )

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            self.component_list,
            doc="Component mole fraction integrated over theta, function of z [-]",
        )
        def y_kz_eq(b, t, z, k):
            return b.y_kz[t, z, k] * b.Flow_z[t, z] == b.Flow_kz[t, z, k]

        def Cp_g_(k, Tg):
            if k == "H2O":
                return (
                    30.09200 * units.kJ / units.mol / units.K
                    + 6.832514 * (Tg / units.K / 1000) * units.kJ / units.mol / units.K
                    + 6.793435
                    * (Tg / units.K / 1000) ** 2
                    * units.kJ
                    / units.mol
                    / units.K
                    + -2.534480
                    * (Tg / units.K / 1000) ** 3
                    * units.kJ
                    / units.mol
                    / units.K
                    + 0.082139
                    / (Tg / units.K / 1000) ** 2
                    * units.kJ
                    / units.mol
                    / units.K
                ) / 1000
            elif k == "N2":
                return (
                    28.98641 * units.kJ / units.mol / units.K
                    + 1.853978 * (Tg / units.K / 1000) * units.kJ / units.mol / units.K
                    + -9.647459
                    * (Tg / units.K / 1000) ** 2
                    * units.kJ
                    / units.mol
                    / units.K
                    + 16.63537
                    * (Tg / units.K / 1000) ** 3
                    * units.kJ
                    / units.mol
                    / units.K
                    + 0.000117
                    / (Tg / units.K / 1000) ** 2
                    * units.kJ
                    / units.mol
                    / units.K
                ) / 1000
            elif k == "CO2":
                return (
                    24.99735 * units.kJ / units.mol / units.K
                    + 55.18696 * (Tg / units.K / 1000) * units.kJ / units.mol / units.K
                    + -33.69137
                    * (Tg / units.K / 1000) ** 2
                    * units.kJ
                    / units.mol
                    / units.K
                    + 7.948387
                    * (Tg / units.K / 1000) ** 3
                    * units.kJ
                    / units.mol
                    / units.K
                    + -0.136638
                    / (Tg / units.K / 1000) ** 2
                    * units.kJ
                    / units.mol
                    / units.K
                ) / 1000

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            doc="pure component heat capacities, function of T [kJ/mol/K]",
        )
        def Cp_g(b, t, z, o, k):
            return Cp_g_(k, b.Tg[t, z, o])

        @blk.Expression(
            self.flowsheet().time, blk.z, blk.o, doc="average molecular weight [kg/mol]"
        )
        def AMW(b, t, z, o):
            return sum([b.y[t, z, o, k] * self.MW[k] for k in self.component_list])

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="gas density [kg/m^3]")
        def rhog(b, t, z, o):
            return b.AMW[t, z, o] * b.C_tot[t, z, o]

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas phase mixture heat capacity [kJ/mol/K]",
        )
        def Cp_g_mix(b, t, z, o):
            return sum(
                [b.y[t, z, o, k] * b.Cp_g[t, z, o, k] for k in self.component_list]
            )

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas phase mixture heat capacity [kJ/kg/K]",
        )
        def Cp_g_mix_kg(b, t, z, o):
            return b.Cp_g_mix[t, z, o] / b.AMW[t, z, o]

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas phase mixture viscosity [Pa*s]",
        )
        def mu_mix(b, t, z, o):
            return sum(
                [
                    b.y[t, z, o, k] * self.mu[k] * self.MW[k] ** 0.5
                    for k in self.component_list
                ]
            ) / sum([b.y[t, z, o, k] * self.MW[k] ** 0.5 for k in self.component_list])

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas mixture thermal conductivity [kW/m/K]",
        )
        def k_mix(b, t, z, o):
            return sum([b.y[t, z, o, k] * self.k[k] for k in self.component_list])

        @blk.Constraint(
            self.flowsheet().time, blk.z, blk.o, doc="heat flux equation [kJ/s/m^2]"
        )
        def heat_flux_eq(b, t, z, o):
            return (
                b.heat_flux[t, z, o]
                == b.C_tot[t, z, o]
                * b.Cp_g_mix[t, z, o]
                * b.vel[t, z, o]
                * b.Tg[t, z, o]
            )

        @blk.Constraint(
            self.flowsheet().time, blk.z, blk.o, doc="mole fraction summation"
        )
        def mole_frac_sum(b, t, z, o):
            if blk.CONFIG.gas_flow_direction == "forward":
                if z > 0:
                    return sum([b.y[t, z, o, k] for k in self.component_list]) == 1
                else:
                    return Constraint.Skip
            elif blk.CONFIG.gas_flow_direction == "reverse":
                if z < 1:
                    return sum([b.y[t, z, o, k] for k in self.component_list]) == 1
                else:
                    return Constraint.Skip

        # Dimensionless groups ===
        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="Prandtl number")
        def Pr(b, t, z, o):
            return b.mu_mix[t, z, o] * b.Cp_g_mix_kg[t, z, o] / b.k_mix[t, z, o]

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="Reynolds number")
        def Re(b, t, z, o):
            return b.rhog[t, z, o] * b.vel[t, z, o] * self.dp / b.mu_mix[t, z, o]

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="Schmidt number")
        def Sc(b, t, z, o):
            return b.mu_mix[t, z, o] / (b.rhog[t, z, o] * self.DmCO2)

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="Sherwood number")
        def Sh(b, t, z, o):
            return (
                2.0
                + 0.6
                * smooth_max(0, b.Re[t, z, o]) ** 0.33
                * smooth_max(0, b.Sc[t, z, o]) ** 0.5
            )

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="Nusselt number")
        def Nu(b, t, z, o):
            return (
                2.0
                + 1.1
                * smooth_max(0, b.Re[t, z, o]) ** 0.6
                * smooth_max(0, b.Pr[t, z, o]) ** 0.33
            )

        # =======================

        # Mass/Heat Transfer coefficients ===
        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="Gas-solid heat transfer coefficient  equation [kW/m^2/K]",
        )
        def h_gs(b, t, z, o):
            return b.Nu[t, z, o] * b.k_mix[t, z, o] / self.dp

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="Gas phase film mass transfer coefficient [m/s]",
        )
        def k_f(b, t, z, o):
            return b.Sh[t, z, o] * self.DmCO2 / self.dp

        # ===================================

        # Isotherm Model Equations ===
        def d_1(T):
            return self.d_inf_1 * exp(
                -self.E_1 / (self.R * self.T0) * (self.T0 / T - 1)
            )

        def d_2(T):
            return self.d_inf_2 * exp(
                -self.E_2 / (self.R * self.T0) * (self.T0 / T - 1)
            )

        def d_3(T):
            return self.d_inf_3 * exp(
                -self.E_3 / (self.R * self.T0) * (self.T0 / T - 1)
            )

        def d_4(T):
            return self.d_inf_4 * exp(
                -self.E_4 / (self.R * self.T0) * (self.T0 / T - 1)
            )

        def sigma_1(T):
            return self.X_11 * exp(self.X_21 * (1 / self.T0 - 1 / T))

        def sigma_2(T):
            return self.X_12 * exp(self.X_22 * (1 / self.T0 - 1 / T))

        def ln_pstep1(T):
            return self.ln_P0_1 + (-self.H_step_1 / self.R * (1 / self.T0 - 1 / T))

        def ln_pstep2(T):
            return self.ln_P0_2 + (-self.H_step_2 / self.R * (1 / self.T0 - 1 / T))

        def q_star_1(P, T):
            return self.q_inf_1 * d_1(T) * P / (1 + d_1(T) * P)

        def q_star_2(P, T):
            return self.q_inf_2 * d_2(T) * P / (1 + d_2(T) * P)

        def q_star_3(P, T):
            return self.q_inf_3 * d_3(T) * P / (1 + d_3(T) * P) + d_4(T) * P

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="Partial pressure of CO2 at particle surface [bar] (ideal gas law)",
        )
        def P_surf(b, t, z, o):
            # smooth max operator: max(0, x) = 0.5*(x + (x^2 + eps)^0.5)
            eps = 1e-8
            Cs_r_smooth_max = 0.5 * (
                b.Cs_r[t, z, o]
                + (b.Cs_r[t, z, o] ** 2 + eps * (units.mol / units.m**3) ** 2) ** 0.5
            )
            return Cs_r_smooth_max * self.Rg * b.Ts[t, z, o]
            # return smooth_max(0,m.Cs_r[z, o]) * self.Rg * m.Ts[z, o] #idaes smooth_max doesn't carry units through

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="log(Psurf)")
        def ln_Psurf(b, t, z, o):
            return log(b.P_surf[t, z, o] / units.bar)  # must make dimensionless

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="weighting function term1: (ln_Psurf-ln_Pstep)/sigma",
        )
        def iso_w_term1(b, t, z, o):
            return (b.ln_Psurf[t, z, o] - ln_pstep1(b.Ts[t, z, o])) / sigma_1(
                b.Ts[t, z, o]
            )

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="weighting function term2: (ln_Psurf-ln_Pstep)/sigma",
        )
        def iso_w_term2(b, t, z, o):
            return (b.ln_Psurf[t, z, o] - ln_pstep2(b.Ts[t, z, o])) / sigma_2(
                b.Ts[t, z, o]
            )

        @blk.Expression(
            self.flowsheet().time, blk.z, blk.o, doc="log of weighting function 1"
        )
        def ln_w1(b, t, z, o):
            # return gamma_1*log(exp(m.iso_w_term1[z,o])/(1+exp(m.iso_w_term1[z,o])))
            # return gamma_1*(log(exp(m.iso_w_term1[z,o])) - log(1+exp(m.iso_w_term1[z,o])))
            return self.gamma_1 * (
                b.iso_w_term1[t, z, o] - log(1 + exp(b.iso_w_term1[t, z, o]))
            )

        @blk.Expression(
            self.flowsheet().time, blk.z, blk.o, doc="log of weighting function 2"
        )
        def ln_w2(b, t, z, o):
            # return gamma_2*log(exp(m.iso_w_term2[z,o])/(1+exp(m.iso_w_term2[z,o])))
            # return gamma_2*(log(exp(m.iso_w_term2[z,o])) - log(1+exp(m.iso_w_term2[z,o])))
            return self.gamma_2 * (
                b.iso_w_term2[t, z, o] - log(1 + exp(b.iso_w_term2[t, z, o]))
            )

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="weighting function 1")
        def iso_w1(b, t, z, o):
            return exp(b.ln_w1[t, z, o])

        @blk.Expression(self.flowsheet().time, blk.z, blk.o, doc="weighting function 2")
        def iso_w2(b, t, z, o):
            return exp(b.ln_w2[t, z, o])

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="isotherm loading expression [mol/kg]",
        )
        def qCO2_eq(b, t, z, o):
            return (
                (1 - b.iso_w1[t, z, o]) * q_star_1(b.P_surf[t, z, o], b.Ts[t, z, o])
                + (b.iso_w1[t, z, o] - b.iso_w2[t, z, o])
                * q_star_2(b.P_surf[t, z, o], b.Ts[t, z, o])
                + b.iso_w2[t, z, o] * q_star_3(b.P_surf[t, z, o], b.Ts[t, z, o])
            )

        # ============================

        # Mass transfer coefficient =======================================================
        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="effective diffusion in solids [m^2/s]",
        )
        def Deff(b, t, z, o):
            return self.C1 * b.Ts[t, z, o] ** 0.5

        @blk.Expression(
            self.flowsheet().time, blk.z, blk.o, doc="internal MT coeff. [1/s]"
        )
        def k_I(b, t, z, o):
            return (
                b.R_MT_coeff * (15 * self.ep * b.Deff[t, z, o] / self.rp**2)
                + (1 - b.R_MT_coeff) * 0.001 / units.s
            )

        # Heat of adsorption ==============================================================
        @blk.Expression(
            self.flowsheet().time, blk.z, blk.o, doc="heat of adsorption [kJ/mol]"
        )
        def delH_CO2(b, t, z, o):
            return -(
                self.delH_1
                - (self.delH_1 - self.delH_2)
                * exp(self.delH_a1 * (b.qCO2_eq[t, z, o] - self.delH_b1))
                / (1 + exp(self.delH_a1 * (b.qCO2_eq[t, z, o] - self.delH_b1)))
                - (self.delH_2 - self.delH_3)
                * exp(self.delH_a2 * (b.qCO2_eq[t, z, o] - self.delH_b2))
                / (1 + exp(self.delH_a2 * (b.qCO2_eq[t, z, o] - self.delH_b2)))
            )

        # Mass/heat transfer rates =========================================================
        # flux limiter equation ===
        a1_FL = 0.02
        a2_FL = 0.98
        sig_FL = 0.01

        def FL(z):
            def FL_1(z):
                return exp((z - a1_FL) / sig_FL) / (1 + exp((z - a1_FL) / sig_FL))

            def FL_2(z):
                return exp((z - a2_FL) / sig_FL) / (1 + exp((z - a2_FL) / sig_FL))

            return FL_1(z) - FL_2(z)

        # ========================

        # Mass Transfer Rates ===
        blk.Rs_CO2 = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=0,
            domain=Reals,
            units=units.mol / units.s / units.m**3,
            doc="solids mass transfer rate [mol/s/m^3 bed]",
        )

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="solids mass transfer rate [mol/s/m^3 bed]",
        )
        def Rs_CO2_eq(b, t, z, o):
            flux_lim = FL(z)

            if 0 < z < 1 and 0 < o < 1:
                return (
                    b.Rs_CO2[t, z, o]
                    == flux_lim
                    * b.k_I[t, z, o]
                    * (b.qCO2_eq[t, z, o] - b.qCO2[t, z, o])
                    * (1 - self.eb)
                    * self.rho_sol
                )
            else:
                return b.Rs_CO2[t, z, o] == 0 * units.mol / units.s / units.m**3

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas mass transfer rate [mol/s/m^3 bed]",
        )
        def Rg_CO2(b, t, z, o):
            if 0 < z < 1 and 0 < o < 1:  # no mass transfer at boundaries
                return b.Rs_CO2[t, z, o]  # option 2
            else:
                return 0 * units.mol / units.s / units.m**3

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas and solid phase mass transfer continuity",
        )
        def constr_MTcont(b, t, z, o):
            """
            Mass transfer continuity between the gas and solid phase. Used to calculate
            Csurf which sets driving force for gas phase mass transfer. A couple options
            for how to write this.

            If m.Rg_CO2 = m.kf*m.a_s*(m.C['CO2']-m.Cs_r) set as expression, then:
                m.Rg_CO2[z,o] == m.Rs_CO2[z,o]

            If m.Rg_CO2 = m.Rs_CO2 set as expression, then:
                m.Rg_CO2[z,o] == m.k_f[z,o]*m.a_s*(m.C['CO2',z,o]-m.Cs_r[z,o])

            """
            return b.Cs_r[t, z, o] == b.C[t, z, o, "CO2"] - b.Rg_CO2[t, z, o] / (
                b.k_f[t, z, o] * self.a_s
            )  # option 2b

        # ========================

        # Heat transfer rates ====
        blk.Q_gs = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=0,
            domain=Reals,
            units=units.kJ / units.s / units.m**3,
            doc="Gas-to-solid heat transfer rate [kW/m^3 bed or kJ/s/m^3 bed]",
        )

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="Gas-to-solid heat transfer rate [kW/m^3 bed or kJ/s/m^3 bed]",
        )
        def Q_gs_eq(b, t, z, o):
            flux_lim = FL(z)

            if 0 < z < 1 and 0 < o < 1:  # no heat transfer at boundaries
                return b.Q_gs[t, z, o] == flux_lim * b.R_HT_gs * b.h_gs[
                    t, z, o
                ] * self.a_s * (b.Ts[t, z, o] - b.Tg[t, z, o])
            else:
                return b.Q_gs[t, z, o] == 0

        blk.Q_ghx = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=0,
            domain=Reals,
            units=units.kJ / units.s / units.m**3,
            doc="Gas-to-HX heat transfer rate [kW/m^3 bed or kJ/s/m^3 bed]",
        )

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="Gas-to-HX heat transfer rate [kW/m^3 bed or kJ/s/m^3 bed]",
        )
        def Q_ghx_eq(b, t, z, o):
            flux_lim = FL(z)

            if 0 < z < 1 and 0 < o < 1:  # no heat transfer at boundaries
                return b.Q_ghx[t, z, o] == flux_lim * b.R_HT_ghx * b.hgx * b.a_ht * (
                    b.Tg[t, z, o] - b.Tx[t]
                )
            else:
                return b.Q_ghx[t, z, o] == 0

        blk.Q_delH = Var(
            self.flowsheet().time,
            blk.z,
            blk.o,
            initialize=0,
            domain=Reals,
            units=units.kJ / units.s / units.m**3,
            doc="adsorption/desorption heat rate [kJ/s/m^3 bed]",
        )

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="adsorption/desorption heat rate [kJ/s/m^3 bed]",
        )
        def Q_delH_eq(b, t, z, o):
            return (
                b.Q_delH[t, z, o] == b.R_delH * b.delH_CO2[t, z, o] * b.Rs_CO2[t, z, o]
            )

        # =====================================================================================

        # PDE equations =======================================================================
        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            doc="gas phase species balance PDE [mol/m^3 bed/s]",
        )
        def pde_gasMB(b, t, z, o, k):
            if blk.CONFIG.gas_flow_direction == "forward":
                if 0 < z < 1:
                    if k == "CO2":
                        return (
                            b.dFluxdz[t, z, o, k]
                            == (-b.Rg_CO2[t, z, o] * b.R_MT_gas) * self.L
                        )
                    else:
                        return b.dFluxdz[t, z, o, k] == 0
                if z == 1:  # at exit of column, dFluxdz=0
                    return b.dFluxdz[t, z, o, k] == 0
                else:  # no balance at z=0, inlets are specified
                    return Constraint.Skip
            elif blk.CONFIG.gas_flow_direction == "reverse":
                if 0 < z < 1:
                    if k == "CO2":
                        return (
                            -b.dFluxdz[t, z, o, k]
                            == (-b.Rg_CO2[t, z, o] * b.R_MT_gas) * self.L
                        )
                    else:
                        return -b.dFluxdz[t, z, o, k] == 0
                if z == 0:  # at exit of column, dFluxdz=0
                    return -b.dFluxdz[t, z, o, k] == 0
                else:  # no balance at z=0, inlets are specified
                    return Constraint.Skip

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            self.component_list,
            doc="flux equation [mol/m^2 bed/s]",
        )
        def flux_eq(b, t, z, o, k):
            return b.Flux_kzo[t, z, o, k] == b.C[t, z, o, k] * b.vel[t, z, o]

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="solid phase mass balance PDE [mol/m^3 bed/s]",
        )
        def pde_solidMB(b, t, z, o):
            if 0 < o < 1:
                return (1 - self.eb) * self.rho_sol * b.dqCO2do[t, z, o] * self.w[
                    t
                ] == (b.Rs_CO2[t, z, o] * b.R_MT_solid) * (
                    (2 * const.pi * units.radians) * b.theta
                )
            elif o == 1:  # at solids exit, flux is zero
                return b.dqCO2do[t, z, o] == 0
            else:  # no balance at o=0, inlets are specified
                return Constraint.Skip

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="gas phase energy balance PDE [kJ/m^3 bed/s]",
        )
        def pde_gasEB(b, t, z, o):
            if blk.CONFIG.gas_flow_direction == "forward":
                if 0 < z < 1:
                    return (
                        b.dheat_fluxdz[t, z, o]
                        == (b.Q_gs[t, z, o] - b.Q_ghx[t, z, o]) * self.L
                    )
                elif z == 1:
                    return b.dheat_fluxdz[t, z, o] == 0
                else:
                    return Constraint.Skip
            elif blk.CONFIG.gas_flow_direction == "reverse":
                if 0 < z < 1:
                    return (
                        -b.dheat_fluxdz[t, z, o]
                        == (b.Q_gs[t, z, o] - b.Q_ghx[t, z, o]) * self.L
                    )
                elif z == 0:
                    return -b.dheat_fluxdz[t, z, o] == 0
                else:
                    return Constraint.Skip

        @blk.Constraint(
            self.flowsheet().time,
            blk.z,
            blk.o,
            doc="solid phase energy balance PDE [kJ/s/m^3 bed]",
        )
        def pde_solidEB(b, t, z, o):
            if 0 < o < 1:
                return (1 - self.eb) * self.rho_sol * self.Cp_sol * self.w[t] * b.dTsdo[
                    t, z, o
                ] == (-b.Q_gs[t, z, o] - b.Q_delH[t, z, o]) * (
                    (2 * const.pi * units.radians) * b.theta
                )
            elif o == 1:
                return b.dTsdo[t, z, o] == 0
            else:
                return Constraint.Skip

        @blk.Constraint(
            self.flowsheet().time, blk.z, blk.o, doc="Ergun Equation [bar/m]"
        )
        def pde_Ergun(b, t, z, o):
            Pa_to_bar = 1e-5 * units.bar / units.Pa
            RHS = b.R_dP * -(
                Pa_to_bar
                * (150 * b.mu_mix[t, z, o] * ((1 - self.eb) ** 2) / (self.eb**3))
                / self.dp**2
                * b.vel[t, z, o]
                + Pa_to_bar
                * 1.75
                * (1 - self.eb)
                / self.eb**3
                * b.rhog[t, z, o]
                / self.dp
                * b.vel[t, z, o] ** 2
            )
            if blk.CONFIG.gas_flow_direction == "forward":
                if z > 0:
                    return b.dPdz[t, z, o] / self.L == RHS
                else:
                    return Constraint.Skip
            elif blk.CONFIG.gas_flow_direction == "reverse":
                if z < 1:
                    return -b.dPdz[t, z, o] / self.L == RHS
                else:
                    return Constraint.Skip

        # =====================================================================================

        # Boundary Conditions ===
        if blk.CONFIG.gas_flow_direction == "forward":
            inlet_node = 0
            outlet_node = 1
        elif blk.CONFIG.gas_flow_direction == "reverse":
            inlet_node = 1
            outlet_node = 0

        @blk.Constraint(self.flowsheet().time, blk.o, doc="inlet gas temp. B.C. [K]")
        def bc_gastemp_in(b, t, o):
            return b.Tg[t, inlet_node, o] == b.Tg_in[t]

        @blk.Constraint(self.flowsheet().time, blk.o, doc="inlet pressure [bar]")
        def bc_P_in(b, t, o):
            return b.P[t, inlet_node, o] == b.P_in[t]

        @blk.Constraint(self.flowsheet().time, doc="inlet flow B.C. [mol/s]")
        def bc_flow_in(b, t):
            return b.Flow_z[t, inlet_node] == b.F_in[t]

        @blk.Constraint(
            self.flowsheet().time,
            blk.o,
            self.component_list,
            doc="inlet mole fraction B.C. [-]",
        )
        def bc_y_in(b, t, o, k):
            return b.y[t, inlet_node, o, k] == b.y_in[t, k]

        @blk.Integral(
            self.flowsheet().time, blk.o, wrt=blk.o, doc="outlet gas enthalpy [kJ/mol]"
        )
        def Hg_out(b, t, o):
            return b.heat_flux[t, outlet_node, o] * b.A_b

        @blk.Constraint(self.flowsheet().time, doc="Outlet flow B.C.")
        def bc_flow_out(b, t):
            return b.F_out[t] == b.Flow_z[t, outlet_node]

        @blk.Constraint(
            self.flowsheet().time, self.component_list, doc="Outlet mole fraction B.C."
        )
        def bc_y_out(b, t, k):
            return b.y_out[t, k] == b.y_kz[t, outlet_node, k]

        @blk.Constraint(self.flowsheet().time, blk.o, doc="outlet pressure B.C. [bar]")
        def bc_P_out(b, t, o):
            return b.P[t, outlet_node, o] == b.P_out[t]

        @blk.Expression(
            self.flowsheet().time, doc="outlet gas heat capacity [kJ/mol/K]"
        )
        def Cp_g_out(b, t):
            return sum(
                [b.y_out[t, k] * Cp_g_(k, b.Tg_out[t]) for k in self.component_list]
            )

        @blk.Constraint(
            self.flowsheet().time, doc="eq. for calculating outlet gas temperature"
        )
        def Tg_out_eq(b, t):
            return b.Hg_out[t] / b.F_out[t] == b.Cp_g_out[t] * b.Tg_out[t]

        # ======================================================================

        # Metrics ==============================================================
        @blk.Expression(
            self.flowsheet().time, blk.z, doc="inlet solids loading B.C. [mol/kg]"
        )
        def qCO2_in(b, t, z):
            return b.qCO2[t, z, 0]

        @blk.Expression(self.flowsheet().time, blk.z, doc="inlet solids temp. [K]")
        def Ts_in(b, t, z):
            return b.Ts[t, z, 0]

        @blk.Expression(self.flowsheet().time, doc="CO2 captured [mol/s]")
        def delta_CO2(b, t):
            return b.F_in[t] * b.y_in[t, "CO2"] - b.F_out[t] * b.y_out[t, "CO2"]

        blk.CO2_capture = Var(
            self.flowsheet().time,
            initialize=0.5,
            domain=Reals,
            doc="CO2 capture fraction",
        )

        @blk.Constraint(self.flowsheet().time, doc="CO2 capture fraction")
        def CO2_capture_eq(b, t):
            return (
                # m.CO2_capture * m.F_in == m.F_in - m.F_out * m.y_out["CO2"] / m.y_in["CO2"]
                b.CO2_capture[t] * b.F_in[t] * b.y_in[t, "CO2"]
                == b.F_in[t] * b.y_in[t, "CO2"] - b.F_out[t] * b.y_out[t, "CO2"]
            )

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            blk.o,
            wrt=blk.o,
            doc="Gas to HX heat transfer integrated over theta, function of z [kW/m^3 bed]",
        )
        def Q_ghx_z(b, t, z, o):
            return b.Q_ghx[t, z, o]

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            wrt=blk.z,
            doc="Gas to HX heat transfer integrated over z, [kW/m^3 bed]",
        )
        def Q_ghx_tot(b, t, z):
            return b.Q_ghx_z[t, z]

        @blk.Expression(doc="section bed volume [m^3 bed]")
        def bed_vol_section(b):
            return const.pi * (self.D / 2) ** 2 * self.L * (1 - b.Hx_frac) * b.theta

        @blk.Expression(self.flowsheet().time, doc="Total heat transfer to HX [kW]")
        def Q_ghx_tot_kW(b, t):
            return b.Q_ghx_tot[t] * b.bed_vol_section

        # ==================================================================================

        # Mass and energy balance checks ===================================================
        @blk.Expression(
            self.flowsheet().time,
            self.component_list,
            doc="component gas flow in for MB check [mol/s]",
        )
        def g_k_in_MB(b, t, k):
            return b.F_in[t] * b.y_in[t, k]

        @blk.Expression(
            self.flowsheet().time,
            self.component_list,
            doc="gas flow out for MB check [mol/s]",
        )
        def g_k_out_MB(b, t, k):
            return b.F_out[t] * b.y_out[t, k]

        @blk.Expression(doc="total bed volume [m^3]")
        def vol_tot(b):
            return const.pi * (self.D / 2) ** 2 * self.L * (1 - b.Hx_frac)

        @blk.Expression(doc="total solids volume [m^3]")
        def vol_solids_tot(b):
            return b.vol_tot * (1 - self.eb)

        @blk.Expression(doc="total solids mass [kg]")
        def mass_solids_tot(b):
            return b.vol_solids_tot * self.rho_sol

        @blk.Expression(self.flowsheet().time, doc="total solids flow [kg/s]")
        def flow_solids_tot(b, t):
            return (
                (b.mass_solids_tot / units.revolution)
                * self.w_rpm[t]
                / (60 * units.sec / units.min)
            )

        @blk.Expression(
            self.flowsheet().time,
            blk.z,
            doc="change in solids loading at each z index [mol/kg]",
        )
        def delta_q(b, t, z):
            return b.qCO2[t, z, 1] - b.qCO2[t, z, 0]

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            wrt=blk.z,
            doc="total flow of adsorbed CO2 at inlet [mol/s]",
        )
        def flow_CO2solids_in(b, t, z):
            return b.qCO2[t, z, 0] * b.flow_solids_tot[t]

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            wrt=blk.z,
            doc="total flow of adsorbed CO2 at outlet [mol/s]",
        )
        def flow_CO2solids_out(b, t, z):
            return b.qCO2[t, z, 1] * b.flow_solids_tot[t]

        @blk.Expression(
            self.flowsheet().time,
            self.component_list,
            doc="% mass balance error for each component",
        )
        def MB_error(b, t, k):
            if k == "CO2":
                return (
                    (b.flow_CO2solids_out[t] + b.g_k_out_MB[t, "CO2"])
                    / (b.flow_CO2solids_in[t] + b.g_k_in_MB[t, "CO2"])
                    - 1
                ) * 100
            else:
                return (
                    b.g_k_out_MB[t, k] / b.g_k_in_MB[t, k] - 1
                ) * 100  # (out-in)/in*100 or (out/in-1)*100

        # ==================================================================================

        # Miscellaneous ====================================================================
        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            blk.o,
            wrt=blk.z,
            doc="solids CO2 loading integrated over z, function of theta [mol/kg]",
        )
        def qCO2_o(b, t, z, o):
            return b.qCO2[t, z, o]

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            blk.o,
            wrt=blk.z,
            doc="solids temperature integrated over z, function of theta [K]",
        )
        def Ts_o(b, t, z, o):
            return b.Ts[t, z, o]

        @blk.Integral(
            self.flowsheet().time,
            blk.z,
            blk.o,
            wrt=blk.o,
            doc="gas temperature integrated over theta, function of z [K]",
        )
        def Tg_z(b, t, z, o):
            return b.Tg[t, z, o]

        # ==============================================================================

        # DAE Transformations ==========================================================
        if self.CONFIG.z_disc_method == "Collocation":
            z_discretizer = TransformationFactory("dae.collocation")
            z_discretizer.apply_to(
                blk,
                wrt=blk.z,
                nfe=self.CONFIG.z_nfe,
                ncp=self.CONFIG.z_Collpoints,
                scheme="LAGRANGE-RADAU",
            )
        elif self.CONFIG.z_disc_method == "Finite Difference":
            z_discretizer = TransformationFactory("dae.finite_difference")
            if blk.CONFIG.gas_flow_direction == "forward":
                z_discretizer.apply_to(
                    blk, wrt=blk.z, nfe=self.CONFIG.z_nfe, scheme="BACKWARD"
                )
            elif blk.CONFIG.gas_flow_direction == "reverse":
                z_discretizer.apply_to(
                    blk, wrt=blk.z, nfe=self.CONFIG.z_nfe, scheme="FORWARD"
                )
        elif self.CONFIG.z_disc_method == "Finite Volume":
            z_discretizer = TransformationFactory("dae.finite_volume")
            if blk.CONFIG.gas_flow_direction == "forward":
                z_discretizer.apply_to(
                    blk,
                    wrt=blk.z,
                    nfv=self.CONFIG.z_nfe,
                    scheme="WENO3",
                    flow_direction=1,
                )
            elif blk.CONFIG.gas_flow_direction == "reverse":
                z_discretizer.apply_to(
                    blk,
                    wrt=blk.z,
                    nfv=self.CONFIG.z_nfe,
                    scheme="WENO3",
                    flow_direction=-1,
                )

        if blk.CONFIG.o_disc_method == "Collocation":
            o_discretizer = TransformationFactory("dae.collocation")
            o_discretizer.apply_to(
                blk,
                wrt=blk.o,
                nfe=blk.CONFIG.o_nfe,
                ncp=blk.CONFIG.o_Collpoints,
            )
        elif blk.CONFIG.o_disc_method == "Finite Difference":
            o_discretizer = TransformationFactory("dae.finite_difference")
            o_discretizer.apply_to(blk, wrt=blk.o, nfe=blk.CONFIG.o_nfe)
        elif blk.CONFIG.o_disc_method == "Finite Volume":
            o_discretizer = TransformationFactory("dae.finite_volume")
            o_discretizer.apply_to(
                blk,
                wrt=blk.o,
                nfv=blk.CONFIG.o_nfe,
                scheme="WENO3",
                flow_direction=1,
            )
        # ==============================================================================

        # initializing variables =======================================================
        for t in self.flowsheet().time:
            for z in blk.z:
                for o in blk.o:
                    for k in self.component_list:
                        blk.y[t, z, o, k] = blk.C_in[t, k]() / blk.C_tot[t, z, o]()
                        blk.Flux_kzo[t, z, o, k] = blk.C_in[t, k]() * blk.vel[t, z, o]()

            # scaling factors ================================
            iscale.set_scaling_factor(blk.bc_P_in, 10)
            iscale.set_scaling_factor(blk.bc_y_out[t, "CO2"], 25)
            iscale.set_scaling_factor(
                blk.bc_y_out[t, "H2O"], 1 / value(blk.y_in[t, "H2O"])
            )
            iscale.set_scaling_factor(
                blk.bc_y_out[t, "N2"], 1 / value(blk.y_in[t, "N2"])
            )
            iscale.set_scaling_factor(blk.bc_P_out, 10)
            iscale.set_scaling_factor(blk.Tg_out[t], 1e-2)
            iscale.set_scaling_factor(
                blk.y_out[t, "H2O"], 1 / value(blk.y_in[t, "H2O"])
            )
            iscale.set_scaling_factor(blk.y_out[t, "N2"], 1 / value(blk.y_in[t, "N2"]))
            iscale.set_scaling_factor(blk.y_out[t, "CO2"], 25)
            iscale.set_scaling_factor(blk.Tx[t], 1e-2)
            iscale.set_scaling_factor(blk.theta, 100)
            iscale.set_scaling_factor(blk.Hg_out[t], 1e-3)
            iscale.set_scaling_factor(blk.F_in[t], 0.001)
            iscale.set_scaling_factor(blk.F_out[t], 0.001)
            iscale.set_scaling_factor(blk.bc_flow_in[t], 0.001)
            iscale.set_scaling_factor(blk.bc_flow_out[t], 0.001)

            for z in blk.z:
                iscale.set_scaling_factor(
                    blk.y_kz[t, z, "N2"], 1 / value(blk.y_in[t, "N2"])
                )
                iscale.set_scaling_factor(blk.y_kz[t, z, "CO2"], 25)
                iscale.set_scaling_factor(
                    blk.y_kz[t, z, "H2O"], 1 / value(blk.y_in[t, "H2O"])
                )
                iscale.set_scaling_factor(
                    blk.y_kz_eq[t, z, "N2"], 0.1 / value(blk.y_in[t, "N2"])
                )
                iscale.set_scaling_factor(blk.y_kz_eq[t, z, "CO2"], 2.5)
                iscale.set_scaling_factor(
                    blk.y_kz_eq[t, z, "H2O"], 0.1 / value(blk.y_in[t, "H2O"])
                )
                iscale.set_scaling_factor(blk.Flow_z[t, z], 0.001)
                iscale.set_scaling_factor(blk.Flow_z_eq[t, z], 0.001)
                for o in blk.o:
                    iscale.set_scaling_factor(blk.vel[t, z, o], 10)
                    iscale.set_scaling_factor(blk.qCO2[t, z, o], 10)
                    iscale.set_scaling_factor(blk.Tg[t, z, o], 1e-2)
                    iscale.set_scaling_factor(blk.Ts[t, z, o], 1e0)
                    iscale.set_scaling_factor(blk.P[t, z, o], 10)
                    iscale.set_scaling_factor(blk.flux_eq[t, z, o, "CO2"], 25)
                    iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "CO2"], 25)
                    iscale.set_scaling_factor(
                        blk.flux_eq[t, z, o, "H2O"], 1 / value(blk.y_in[t, "H2O"])
                    )
                    iscale.set_scaling_factor(
                        blk.Flux_kzo[t, z, o, "H2O"], 1 / value(blk.y_in[t, "H2O"])
                    )
                    iscale.set_scaling_factor(blk.heat_flux_eq[t, z, o], 0.1)
                    iscale.set_scaling_factor(blk.heat_flux[t, z, o], 0.05)
                    iscale.set_scaling_factor(
                        blk.y[t, z, o, "H2O"], 1 / value(blk.y_in[t, "H2O"])
                    )
                    iscale.set_scaling_factor(
                        blk.y[t, z, o, "N2"], 1 / value(blk.y_in[t, "N2"])
                    )
                    iscale.set_scaling_factor(blk.y[t, z, o, "CO2"], 25)
                    iscale.set_scaling_factor(blk.Cs_r[t, z, o], 2.5)
                    iscale.set_scaling_factor(blk.constr_MTcont[t, z, o], 2.5)

                    if o == 0 or o == 1:
                        iscale.set_scaling_factor(blk.flux_eq[t, z, o, "CO2"], 1e1)
                        iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "CO2"], 1e1)
                        iscale.set_scaling_factor(blk.flux_eq[t, z, o, "H2O"], 1e1)
                        iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "H2O"], 1e1)

                    if 0 < z < 1 and 0 < o < 1:
                        iscale.set_scaling_factor(blk.dqCO2do[t, z, o], 1e-2)
                        iscale.set_scaling_factor(blk.dqCO2do_disc_eq[t, z, o], 1e-2)
                        iscale.set_scaling_factor(blk.pde_gasEB[t, z, o], 1e0)
                        iscale.set_scaling_factor(blk.pde_solidEB[t, z, o], 1e2)
                        iscale.set_scaling_factor(blk.pde_solidMB[t, z, o], 1e-3)
                        iscale.set_scaling_factor(blk.dheat_fluxdz[t, z, o], 1e-2)
                        iscale.set_scaling_factor(blk.dTsdo[t, z, o], 1e-1)
                        iscale.set_scaling_factor(blk.dTsdo_disc_eq[t, z, o], 1e-1)
                        iscale.set_scaling_factor(blk.pde_gasMB[t, z, o, "CO2"], 100)
                        iscale.set_scaling_factor(blk.Q_gs_eq[t, z, o], 1)
                        iscale.set_scaling_factor(blk.Q_gs[t, z, o], 0.01)
                        iscale.set_scaling_factor(blk.Q_delH[t, z, o], 0.01)
                        iscale.set_scaling_factor(blk.Q_delH_eq[t, z, o], 0.01)
                        iscale.set_scaling_factor(blk.Rs_CO2[t, z, o], 0.5)
                        iscale.set_scaling_factor(blk.Rs_CO2_eq[t, z, o], 1)

                    if blk.CONFIG.gas_flow_direction == "forward":
                        if z > 0:
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "CO2"], 0.4
                            )
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "H2O"],
                                10 * value(blk.y_in[t, "H2O"]),
                            )
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "N2"], 0.1
                            )
                            iscale.set_scaling_factor(blk.dPdz[t, z, o], 10)
                            iscale.set_scaling_factor(blk.dPdz_disc_eq[t, z, o], 10)
                            iscale.set_scaling_factor(blk.pde_Ergun[t, z, o], 100)
                            iscale.set_scaling_factor(
                                blk.dheat_fluxdz_disc_eq[t, z, o], 1e-2
                            )
                            iscale.set_scaling_factor(blk.dFluxdz[t, z, o, "CO2"], 0.4)
                            iscale.set_scaling_factor(
                                blk.dFluxdz[t, z, o, "H2O"],
                                10 * value(blk.y_in[t, "H2O"]),
                            )
                            iscale.set_scaling_factor(blk.mole_frac_sum[t, z, o], 100)

                        if z == 1:
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "CO2"], 0.1
                            )
                            iscale.set_scaling_factor(blk.dFluxdz[t, z, o, "CO2"], 0.1)
                            iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "CO2"], 1)
                            iscale.set_scaling_factor(blk.flux_eq[t, z, o, "CO2"], 1)
                            iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "H2O"], 1)
                            iscale.set_scaling_factor(blk.flux_eq[t, z, o, "H2O"], 1)

                        if z == 0:
                            iscale.set_scaling_factor(
                                blk.y[t, z, o, "CO2"], 1 / value(blk.y_in[t, "CO2"])
                            )
                    elif blk.CONFIG.gas_flow_direction == "reverse":
                        if z < 1:
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "CO2"], 0.5
                            )
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "H2O"], 0.5
                            )
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "N2"], 0.1
                            )
                            iscale.set_scaling_factor(blk.dPdz[t, z, o], 10)
                            iscale.set_scaling_factor(blk.dPdz_disc_eq[t, z, o], 10)
                            iscale.set_scaling_factor(blk.pde_Ergun[t, z, o], 100)
                            iscale.set_scaling_factor(
                                blk.dheat_fluxdz_disc_eq[t, z, o], 1e-2
                            )
                            iscale.set_scaling_factor(blk.dFluxdz[t, z, o, "CO2"], 0.5)
                            iscale.set_scaling_factor(blk.dFluxdz[t, z, o, "H2O"], 0.5)
                            iscale.set_scaling_factor(blk.mole_frac_sum[t, z, o], 100)

                        if z == 0:
                            iscale.set_scaling_factor(
                                blk.dFluxdz_disc_eq[t, z, o, "CO2"], 0.1
                            )
                            iscale.set_scaling_factor(blk.dFluxdz[t, z, o, "CO2"], 0.1)
                            iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "CO2"], 1)
                            iscale.set_scaling_factor(blk.flux_eq[t, z, o, "CO2"], 1)
                            iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "H2O"], 1)
                            iscale.set_scaling_factor(blk.flux_eq[t, z, o, "H2O"], 1)

                        if z == 1:
                            iscale.set_scaling_factor(
                                blk.y[t, z, o, "CO2"], 1 / value(blk.y_in[t, "CO2"])
                            )

            for o in blk.o:
                iscale.set_scaling_factor(blk.bc_gastemp_in[t, o], 1e-2)
                iscale.set_scaling_factor(
                    blk.bc_y_in[t, o, "CO2"], 1 / value(y_in[t, "CO2"])
                )
                iscale.set_scaling_factor(
                    blk.bc_y_in[t, o, "H2O"], 1 / value(y_in[t, "H2O"])
                )
                iscale.set_scaling_factor(
                    blk.bc_y_in[t, o, "N2"], 1 / value(y_in[t, "N2"])
                )

            if initial_guesses == "desorption":
                iscale.set_scaling_factor(blk.CO2_capture[t], 1e-4)
                iscale.set_scaling_factor(blk.CO2_capture_eq[t], 1e-4)
                iscale.set_scaling_factor(blk.F_in[t], 1e-2)
                iscale.set_scaling_factor(blk.F_out[t], 1e-2)
                iscale.set_scaling_factor(blk.bc_flow_in[t], 1e-2)
                iscale.set_scaling_factor(blk.bc_flow_out[t], 1e-2)
                for z in blk.z:
                    iscale.set_scaling_factor(blk.Flow_z[t, z], 1e-2)
                    iscale.set_scaling_factor(blk.Flow_z_eq[t, z], 1e-2)
                    iscale.set_scaling_factor(
                        blk.y_kz[t, z, "N2"], 0.1 / value(blk.y_in[t, "N2"])
                    )
                    iscale.set_scaling_factor(
                        blk.y_kz_eq[t, z, "N2"], 0.01 / value(blk.y_in[t, "N2"])
                    )
                    for o in blk.o:
                        iscale.set_scaling_factor(blk.flux_eq[t, z, o, "N2"], 1e1)
                        iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "N2"], 1e1)
                        iscale.set_scaling_factor(
                            blk.y[t, z, o, "N2"], 0.1 / value(blk.y_in[t, "N2"])
                        )
                        iscale.set_scaling_factor(blk.flux_eq[t, z, o, "CO2"], 2.5)
                        iscale.set_scaling_factor(blk.Flux_kzo[t, z, o, "CO2"], 2.5)

                        if o == 0 or o == 1:
                            iscale.set_scaling_factor(blk.flux_eq[t, z, o, "H2O"], 1e-1)
                            iscale.set_scaling_factor(
                                blk.Flux_kzo[t, z, o, "H2O"], 1e-1
                            )

        # =================================================================================
        # fixing solid inlet variables ===================================================
        for t in self.flowsheet().time:
            for z in blk.z:
                blk.Ts[t, z, 0].fix()
                blk.qCO2[t, z, 0].fix()
        # =================================================================================


# Creating upper level RPB block
def RotaryPackedBed():

    # blk.time = Set(initialize=[0], doc="time domain [s]")

    # Initial/Inlet/Outlet Values

    # =========================== Dimensions =======================================

    # blk.D.fix()

    # blk.L.fix()

    # blk.w_rpm.fix()

    # =========================== Solids Properties ================================

    # Isotherm Parameter values

    # Mass transfer parameters

    # heat of adsorption ===

    # ==============================================================================

    # ========================== gas properties ====================================

    # blk.gas_props = FlueGasParameterBlock(components=blk.component_list)

    # ==============================================================================

    return None


def add_single_section_equations(
    RPB, section_name, gas_flow_direction="forward", initial_guesses="Adsorption"
):

    # blk.theta.fix()

    # embedded heat exchanger and area calculations

    # ============================ Gas Inlet =======================================

    # =========================== Gas Outlet =======================================

    # ======================== Heat exchanger ======================================

    # Variable declaration
    # ============================== Gas Phase =====================================

    # ========================= Solids =============================================

    # Initialization factors ===

    # Section 2: Gas Equations, Gas Properties, Dimensionless Groups, and Variables related

    # Dimensionless groups ====

    # ===

    # Mass/Heat Transfer variables

    # Model equations

    # Mass transfer coefficient

    # heat of adsorption

    # mass transfer rates ===

    # flux limiter equation

    # heat transfer rates ===

    # PDE equations, boundary conditions, and model constraints

    # Boundary Conditions ===

    # Metrics ==============
    # ==============

    # Mass and energy balance checks
    # ==============

    # miscellaneous
    # ==============

    # DAE Transformations

    # initializing some variables and setting scaling factors

    # need to do this after discretizer is applied or else the new indices won't get the scaling factor

    # initializing variables
    # =================================================

    # fixing solid inlet variables

    return None


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
                y_CO2[k].append(m.y[0, i, theta[j], "CO2"]())
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
                C_CO2[k].append(m.C[0, i, theta[j], "CO2"]())
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
                C_N2[k].append(m.C[0, i, theta[j], "N2"]())
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
                Tg[k].append(m.Tg[0, i, theta[j]]())
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
                Pg[k].append(m.P[0, i, theta[j]]())
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
                vg[k].append(m.vel[0, i, theta[j]]())
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
                qCO2[k].append(m.qCO2[0, z[j], i]())
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
                Ts[k].append(m.Ts[0, z[j], i]())
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
                y[k].append(m.y[0, z[j], i, "CO2"]())
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
    for k in blk.parent_block().component_list:
        # print error for each component formatted in scientific notation
        print(f"{k} error = {blk.MB_error[0,k]():.3} %")


def homotopy_solve1(blk):
    blk.R_HT_gs = 1e-10
    blk.R_HT_ghx = 1e-10
    blk.R_delH = 1e-10
    blk.R_MT_coeff = 1e-10
    blk.R_dP = 1
    blk.R_MT_gas = 1e-10
    blk.R_MT_solid = 1e-10

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

    blk.R_HT_gs = 1e-10
    blk.R_HT_ghx = 1e-10
    blk.R_delH = 1e-10
    blk.R_MT_coeff = 1e-10
    blk.R_dP = 1
    blk.R_MT_gas = 1e-10
    blk.R_MT_solid = 1e-10

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

    blk.R_HT_gs = 1e-10
    blk.R_HT_ghx = 1e-10
    blk.R_delH = 1e-10
    blk.R_MT_coeff = 1e-10
    blk.R_dP = 1
    blk.R_MT_gas = 1e-10
    blk.R_MT_solid = 1e-10

    blk.R_MT_solid = 1
    blk.R_MT_gas = 1

    print(f"DOF = {degrees_of_freedom(blk)}")

    init_obj.initialization_routine(blk)

    blk.R_MT_coeff = 1
    blk.R_HT_ghx = 1
    blk.R_HT_gs = 1

    init_obj.initialization_routine(blk)

    blk.R_delH = 1

    init_obj.initialization_routine(blk)

    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()


def single_section_init2(blk):
    blk.P_in.fix(1.1)
    blk.Tg_in.fix()
    blk.y_in.fix()
    blk.P_out.fix(1.01325)

    blk.R_HT_gs = 1e-10
    blk.R_HT_ghx = 1e-10
    blk.R_delH = 1e-10
    blk.R_MT_coeff = 1e-10
    blk.R_dP = 1
    blk.R_MT_gas = 1e-10
    blk.R_MT_solid = 1e-10

    # add dummy objective
    blk.obj = Objective(expr=0)

    results = SolverFactory("gams").solve(
        blk,
        tee=True,
        keepfiles=True,
        solver="conopt4",
        tmpdir="temp",
        add_options=["gams_model.optfile=1;"],
    )

    blk.R_MT_solid = 1
    blk.R_MT_gas = 1
    blk.R_MT_coeff = 1

    print(f"DOF = {degrees_of_freedom(blk)}")

    results = SolverFactory("gams").solve(
        blk,
        tee=True,
        keepfiles=True,
        solver="conopt4",
        tmpdir="temp",
        add_options=["gams_model.optfile=1;"],
    )

    blk.R_HT_ghx = 1
    blk.R_HT_gs = 1
    blk.R_delH = 1

    results = SolverFactory("gams").solve(
        blk,
        tee=True,
        keepfiles=True,
        solver="conopt4",
        tmpdir="temp",
        add_options=["gams_model.optfile=1;"],
    )


def full_model_creation(lean_temp_connection=True, configuration="co-current"):
    RPB = RotaryPackedBed()

    if configuration == "co-current":
        add_single_section_equations(
            RPB,
            section_name="ads",
            gas_flow_direction="forward",
            initial_guesses="adsorption",
        )
        add_single_section_equations(
            RPB,
            section_name="des",
            gas_flow_direction="forward",
            initial_guesses="desorption",
        )
    elif configuration == "counter-current":
        add_single_section_equations(
            RPB,
            section_name="ads",
            gas_flow_direction="forward",
            initial_guesses="adsorption",
        )
        add_single_section_equations(
            RPB,
            section_name="des",
            gas_flow_direction="reverse",
            initial_guesses="desorption",
        )

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
        if 0 < z < 1:
            RPB.des.qCO2[0, z, 0].unfix()
            RPB.des.Ts[0, z, 0].unfix()

    # add equality constraint equating inlet desorption loading to outlet adsorption loading. Same for temperature.
    @RPB.Constraint(RPB.time, RPB.des.z)
    def rich_loading_constraint(b, t, z):
        if 0 < z < 1:
            return b.des.qCO2[t, z, 0] == b.ads.qCO2[t, z, 1]
        else:
            return Constraint.Skip

    @RPB.Constraint(RPB.time, RPB.des.z)
    def rich_temp_constraint(b, t, z):
        if 0 < z < 1:
            return b.des.Ts[t, z, 0] == b.ads.Ts[t, z, 1]
        else:
            return Constraint.Skip

    # connect lean stream
    # unfix inlet loading to the adsorption section
    for z in RPB.ads.z:
        if 0 < z < 1:
            RPB.ads.qCO2[0, z, 0].unfix()
            if lean_temp_connection:
                RPB.ads.Ts[0, z, 0].unfix()

    # add equality constraint equating inlet adsorption loading to outlet desorption loading
    @RPB.Constraint(RPB.time, RPB.ads.z)
    def lean_loading_constraint(b, t, z):
        if 0 < z < 1:
            return b.ads.qCO2[t, z, 0] == b.des.qCO2[t, z, 1]
        else:
            return Constraint.Skip

    if lean_temp_connection:

        @RPB.Constraint(RPB.time, RPB.ads.z)
        def lean_temp_constraint(b, t, z):
            if 0 < z < 1:
                return b.ads.Ts[t, z, 0] == b.des.Ts[t, z, 1]
            else:
                return Constraint.Skip

    # these variables are inactive, just fixing them to same value for plotting purposes
    for t in RPB.time:
        RPB.ads.qCO2[t, 0, 0].fix(1)
        RPB.ads.qCO2[t, 1, 0].fix(1)
        RPB.des.qCO2[t, 0, 0].fix(1)
        RPB.des.qCO2[t, 1, 0].fix(1)

        RPB.ads.Ts[t, 0, 0].fix(100 + 273)
        RPB.ads.Ts[t, 1, 0].fix(100 + 273)
        RPB.des.Ts[t, 0, 0].fix(100 + 273)
        RPB.des.Ts[t, 1, 0].fix(100 + 273)

    # add constraint so that the fraction of each section adds to 1
    RPB.des.theta.unfix()  # unfix des side var

    @RPB.Constraint(doc="Theta summation constraint")
    def theta_constraint(b):
        return b.ads.theta + b.des.theta == 1

    # metrics ===============================================
    RPB.steam_enthalpy = Param(
        initialize=2257.92,
        mutable=True,
        units=units.kJ / units.kg,
        doc="saturated steam enthalpy at 1 bar[kJ/kg]",
    )

    @RPB.Expression(RPB.time, doc="Steam energy [kW]")
    def steam_energy(b, t):
        return b.des.F_in[t] * b.des.y_in[t, "H2O"] * b.MW["H2O"] * b.steam_enthalpy

    @RPB.Expression(RPB.time, doc="total thermal energy (steam + HX) [kW]")
    def total_thermal_energy(b, t):
        return b.steam_energy[t] - b.des.Q_ghx_tot_kW[t]

    @RPB.Expression(RPB.time, doc="Energy requirement [MJ/kg CO2]")
    def energy_requirement(b, t):
        return units.convert(
            b.total_thermal_energy[t] / b.ads.delta_CO2[t] / b.MW["CO2"],
            to_units=units.MJ / units.kg,
        )

    @RPB.Expression(RPB.time, doc="Productivity [kg CO2/h/m^3]")
    def productivity(b, t):
        return units.convert(
            b.ads.delta_CO2[t] * b.MW["CO2"] / b.ads.vol_tot,
            to_units=units.kg / units.h / units.m**3,
        )

    # add scaling factors
    iscale.set_scaling_factor(RPB.theta_constraint, 1e2)
    for t in RPB.time:
        for z in RPB.ads.z:
            if 0 < z < 1:
                iscale.set_scaling_factor(RPB.lean_loading_constraint[t, z], 10)
                iscale.set_scaling_factor(RPB.rich_loading_constraint[t, z], 10)

    return RPB


def init_routine_1(blk, homotopy_points=[1]):
    # create Block init object
    init_obj = BlockTriangularizationInitializer()

    init_obj.config.block_solver_call_options = {"tee": True}
    init_obj.config.block_solver_options = {
        # "halt_on_ampl_error": "yes",
        "max_iter": 1000,
        # "bound_push": 1e-22,
        # "mu_init": 1e-3,
    }

    blk.ads.R_MT_gas = 1e-10
    blk.des.R_MT_gas = 1e-10
    blk.ads.R_MT_coeff = 1e-10
    blk.des.R_MT_coeff = 1e-10
    blk.ads.R_HT_ghx = 1e-10
    blk.des.R_HT_ghx = 1e-10
    blk.ads.R_HT_gs = 1e-10
    blk.des.R_HT_gs = 1e-10
    blk.ads.R_delH = 1e-10
    blk.des.R_delH = 1e-10

    # run initialization routine
    print("DOF =", degrees_of_freedom(blk))

    init_obj.initialization_routine(blk)

    for i in homotopy_points:
        print(f"homotopy point {i}")
        blk.ads.R_MT_gas = i
        blk.des.R_MT_gas = i
        blk.ads.R_MT_coeff = i
        blk.des.R_MT_coeff = i
        blk.ads.R_HT_ghx = i
        blk.des.R_HT_ghx = i
        blk.ads.R_HT_gs = i
        blk.des.R_HT_gs = i
        blk.ads.R_delH = i
        blk.des.R_delH = i
        init_obj.initialization_routine(blk)

    print("full solve")

    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 500,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()


def init_routine_2(blk):
    init_obj = BlockTriangularizationInitializer()

    init_obj.config.block_solver_call_options = {"tee": True}
    init_obj.config.block_solver_options = {
        # "halt_on_ampl_error": "yes",
        "max_iter": 500,
    }

    blk.ads.R_MT_gas = 1e-10
    blk.des.R_MT_gas = 1e-10
    blk.ads.R_MT_coeff = 1e-10
    blk.des.R_MT_coeff = 1e-10
    blk.ads.R_HT_ghx = 1e-10
    blk.des.R_HT_ghx = 1e-10
    blk.ads.R_HT_gs = 1e-10
    blk.des.R_HT_gs = 1e-10
    blk.ads.R_delH = 1e-10
    blk.des.R_delH = 1e-10

    # turn on solids mass transfer (with the loadings connected at the rich and lean ends, solids mass transfer has to be turned on or no solution exists)
    blk.ads.R_MT_solid = 1
    blk.des.R_MT_solid = 1

    # run initialization routine

    init_obj.initialization_routine(blk)

    solver = SolverFactory("ipopt")
    solver.options = {
        "max_iter": 1000,
        "bound_push": 1e-22,
        "halt_on_ampl_error": "yes",
    }
    solver.solve(blk, tee=True).write()

    variables_list = [
        blk.ads.R_HT_gs,
        blk.des.R_HT_gs,
        blk.ads.R_HT_ghx,
        blk.des.R_HT_ghx,
        blk.ads.R_delH,
        blk.des.R_delH,
        blk.ads.R_MT_coeff,
        blk.des.R_MT_coeff,
        blk.ads.R_MT_gas,
        blk.des.R_MT_gas,
    ]

    targets_list = [
        1,
        1,
        1,
        1,
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


def report(blk):
    items = [
        blk.L,
        blk.D,
        blk.w_rpm[0],
        blk.ads.theta,
        blk.des.theta,
        blk.ads.P_in[0],
        blk.ads.P_out[0],
        blk.ads.F_in[0],
        blk.ads.Tg_in[0],
        blk.ads.Tx[0],
        blk.des.P_in[0],
        blk.des.P_out[0],
        blk.des.F_in[0],
        blk.des.Tg_in[0],
        blk.des.Tx[0],
        blk.ads.CO2_capture[0],
        blk.energy_requirement[0],
        blk.productivity[0],
    ]

    names = []
    values = []
    fixed = []
    lb = []
    ub = []
    docs = []
    for item in items:
        names.append(item.to_string())
        values.append(item())
        if item.ctype != Var:
            fixed.append("N/A")
            lb.append("N/A")
            ub.append("N/A")
        else:
            fixed.append(item.fixed)
            lb.append(item.lb)
            ub.append(item.ub)
        docs.append(item.parent_component().doc)

    report_df = pd.DataFrame(
        data={
            "Value": values,
            "Doc": docs,
            "Fixed": fixed,
            "Lower Bound": lb,
            "Upper Bound": ub,
        },
        index=names,
    )

    indexed_items = [
        blk.ads.inlet.y_in,
        blk.ads.outlet.y_out,
    ]

    names = []
    values = []
    docs = []
    fixed = []
    lb = []
    ub = []
    for item in indexed_items:
        names += [item[k].to_string() for k in item.keys()]
        values += [item[k]() for k in item.keys()]
        docs += [item.doc for k in item.keys()]
        fixed += [item[k].fixed for k in item.keys()]
        lb += [item[k].lb for k in item.keys()]
        ub += [item[k].ub for k in item.keys()]

    report_indexed_df = pd.DataFrame(
        data={
            "Value": values,
            "Doc": docs,
            "Fixed": fixed,
            "Lower Bound": lb,
            "Upper Bound": ub,
        },
        index=names,
    )

    report_df = pd.concat([report_df, report_indexed_df])

    return report_df


def full_contactor_plotting(blk, save_option=False):
    z = list(blk.ads.z)
    theta = list(blk.ads.o)

    theta_total_norm = [j * blk.ads.theta() for j in blk.ads.o] + [
        j * blk.des.theta() + blk.ads.theta() for j in blk.des.o
    ][1:]

    z_query = [0.05, 0.25, 0.5, 0.75, 0.95]
    z_nodes = [blk.ads.z.find_nearest_index(z) for z in z_query]

    theta_query = [0.01, 0.05, 0.3, 0.5, 0.8]
    theta_nodes = [blk.ads.o.find_nearest_index(o) for o in theta_query]

    # Solids Loading
    qCO2_ads = [[], [], [], [], []]
    qCO2_des = [[], [], [], [], []]
    qCO2_total = [[], [], [], [], []]
    k = 0
    for j in z_nodes:
        for i in theta:
            qCO2_ads[k].append(blk.ads.qCO2[0, z[j], i]())
            qCO2_des[k].append(blk.des.qCO2[0, z[j], i]())
        k += 1

    for k in range(len(z_nodes)):
        qCO2_total[k] = qCO2_ads[k] + qCO2_des[k][1:]

    qCO2_avg = [blk.ads.qCO2_o[0, o]() for o in blk.ads.o] + [
        blk.des.qCO2_o[0, o]() for o in blk.des.o
    ][1:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Rotational Distance [-]", fontsize=16)
    ax.set_ylabel("CO$_{2}$ Loading [mol/kg]", fontsize=16)
    # ax.set_title('Adsorption CO$_{2}$ Loading')
    for i in range(len(z_nodes)):
        ax.plot(
            theta_total_norm,
            qCO2_total[i],
            "-o",
            label="z=" + str(round(z[z_nodes[i]], 3)),
        )
    ax.plot(theta_total_norm, qCO2_avg, "--", label="Averaged")
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
            Ts_ads[k].append(blk.ads.Ts[0, z[j], i]())
            Ts_des[k].append(blk.des.Ts[0, z[j], i]())
        k += 1

    for k in range(len(z_nodes)):
        Ts_total[k] = Ts_ads[k] + Ts_des[k][1:]

    Ts_avg = [blk.ads.Ts_o[0, o]() for o in blk.ads.o] + [
        blk.des.Ts_o[0, o]() for o in blk.des.o
    ][1:]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Rotational Distance [-]", fontsize=16)
    ax.set_ylabel("Solids Temperature [K]", fontsize=16)
    # ax.set_title('Adsorption CO$_{2}$ Loading')
    for i in range(len(z_nodes)):
        ax.plot(
            theta_total_norm,
            Ts_total[i],
            "-o",
            label="z=" + str(round(z[z_nodes[i]], 3)),
        )
    ax.plot(theta_total_norm, Ts_avg, "--", label="Averaged")
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
            y_CO2[k].append(blk.ads.y[0, i, theta[j], "CO2"]())
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
    ax.plot(z, [blk.ads.y_kz[0, j, "CO2"]() for j in z], "--", label="Averaged")
    ax.legend()

    if save_option:
        fig.savefig("CO2_molefraction_ads.png", dpi=300)

    # Desorber Gas phase CO2 mole fraction
    y_CO2 = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            y_CO2[k].append(blk.des.y[0, i, theta[j], "CO2"]())
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
    ax.plot(z, [blk.des.y_kz[0, j, "CO2"]() for j in z], "--", label="Averaged")
    ax.legend()

    if save_option:
        fig.savefig("CO2_molefraction_des.png", dpi=300)

    # Adsorber Gas Temperature
    Tg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Tg[k].append(blk.ads.Tg[0, i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas Temperature, Adsorber [K]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, Tg[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.plot(z, [blk.ads.Tg_z[0, j]() for j in z], "--", label="Averaged")
    ax.axhline(
        y=blk.ads.Tx[0](),
        xmin=0,
        xmax=1,
        color="black",
        label="Embedded Heat Exchanger Temp [K]",
    )
    ax.legend()

    if save_option:
        fig.savefig("GasTemp_ads.png", dpi=300)

    # Desorber Gas Temperature
    Tg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Tg[k].append(blk.des.Tg[0, i, theta[j]]())
        k += 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Normalized Axial distance", fontsize=16)
    ax.set_ylabel("Gas Temperature, Desorber [K]", fontsize=16)
    # ax.set_title('Adsorption gas phase CO$_{2}$')
    for i in range(len(theta_nodes)):
        ax.plot(z, Tg[i], "-o", label="theta=" + str(round(theta[theta_nodes[i]], 3)))
    ax.plot(z, [blk.des.Tg_z[0, j]() for j in z], "--", label="Averaged")
    ax.axhline(
        y=blk.des.Tx[0](),
        xmin=0,
        xmax=1,
        color="black",
        label="Embedded Heat Exchanger Temp [K]",
    )
    ax.legend()

    if save_option:
        fig.savefig("GasTemp_des.png", dpi=300)

    # Adsorber Gas Pressure
    Pg = [[], [], [], [], []]
    k = 0
    for j in theta_nodes:
        for i in z:
            Pg[k].append(blk.ads.P[0, i, theta[j]]())
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
            Pg[k].append(blk.des.P[0, i, theta[j]]())
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
            vel[k].append(blk.ads.vel[0, i, theta[j]]())
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
            vel[k].append(blk.des.vel[0, i, theta[j]]())
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
