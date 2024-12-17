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
Utility functions for RPB models
"""
import pandas as pd

from pyomo.environ import Var


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
