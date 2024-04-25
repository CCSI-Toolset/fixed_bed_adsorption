"""
Measurement optimization tool 
@University of Notre Dame
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from greybox_generalize import LogDetModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
)
from enum import Enum
from dataclasses import dataclass
import pickle


class CovarianceStructure(Enum):
    """Covariance definition
    if identity: error covariance matrix is an identity matrix
    if variance: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
        Shape: Sum(Nt)
    if time_correlation: a 3D numpy array, each element is the error covariances
        This option assumes covariances not between measurements, but between timepoints for one measurement
        Shape: Nm * (Nt_m * Nt_m)
    if measure_correlation: a 2D numpy array, covariance matrix for a single time steps
        This option assumes the covariances between measurements at the same timestep in a time-invariant way
        Shape: Nm * Nm
    if time_measure_correlation: a 2D numpy array, covariance matrix for the flattened measurements
        Shape: sum(Nt) * sum(Nt)
    """

    identity = 0
    variance = 1
    time_correlation = 2
    measure_correlation = 3
    time_measure_correlation = 4

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ObjectiveLib(Enum):
    """
    Objective function library

    if A: minimize the trace of FIM
    if D: minimize the determinant of FIM
    """

    A = 0
    D = 1

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class SensitivityData:
    """"""

    def __init__(self, filename, Nt) -> None:
        """
        Read, convert the format, and sotre the sensitivity information (Jacobian).
        Only process a certain format of CSV file.
        Assume every measurement has the same number of time points: Nt

        Arguments
        ---------
        :param filename: a string of the csv file name
        :param Nt: number of timepoints is needed to split Q for each measurement.
            Assume all measurements have the same number of time points of Nt.

        This csv file should have the following format:
        columns: parameters to be estimated
        (An extra first column is added for index)
        rows: measurement timepoints
        data: Jacobian values

        The csv file example:

        Column index  |  Parameter 1 | Parameter 2 |  ...  | Parameter P
        measurement 1 |    number    |  number     |  ...  | number
        measurement 2 |    number    |  number     |  ...  | number
        ...
        measurement N |    number    |  number     |  ...  | number

        Number according to measurement i, parameter j is
        the gradient of measurement i w.r.t parameter j

        Returns
        ------
        None
        """
        # store the number of time points for each measurement
        self.Nt = Nt
        # read Jacobian from .csv file
        self._read_from_pickle(filename)

    def _read_from_pickle(self, filename):
        """Read Jacobian from pickle file

        Arguments
        ---------
        :param filename: a string of the pickle file name
            This pickle file structure is explained in the class doc string.

        Returns
        ------
        None
        """
        file = open(filename, "rb")
        jacobian_array = pickle.load(file)
        file.close()

        # check if the number of rows in this array is a multiple of Nt
        if len(jacobian_array) % self.Nt != 0:
            raise ValueError(
                "The number of rows in this Jacobian matrix is not a multiple of Nt."
            )

        # infer the number of parameters from the shape of jacobian
        # the first column is the measurement name, so needs to be deleted
        self.n_parameters = len(jacobian_array[0])

        # store the original Jacobian matrix in object
        self.jacobian = jacobian_array

    def get_jac_list(self, static_measurement_idx, dynamic_measurement_idx):
        """Combine Jacobian Q for each measurement to be in one Jacobian Q.
        Through _split_jacobian we get a list of lists for Jacobian Q,
        each list contains an Nt*n_parameters elements, which is the sensitivity matrix Q for measurement m

        We aim to convert this list of arrays to Q, a numpy array containing Jacobian matrix of the shape N_total_m * Np
        Jacobian Q structure:
        [ \partial y1(t1)/ \partial p1, ..., \partial y1(t1)/ \partial pN,
        \partial y1(t2)/ \partial p1, ..., \partial y1(t2)/ \partial pN,
        ...,
        \partial y1(tN)/ \partial p1, ..., \partial y1(tN)/ \partial pN,
        \partial y2(t1)/ \partial p1, ..., \partial y2(t1)/ \partial pN,
        ...,
        \partial yN(tN)/ \partial p1, ..., \partial yN(tN)/ \partial pN,]

        Arguments
        ---------
        :param static_measurement_idx: list of the index for static measurements
        :param dynamic_measurement_idx: list of the index for dynamic measurements

        Returns
        -------
        self.jac: Jacobian information for main class use, a 2D numpy array
        """

        # get the maximum index from index set
        # why not sum up static index number and dynamic index number? because they can overlap
        if not dynamic_measurement_idx:
            max_measure_idx = max(static_measurement_idx)
        elif not static_measurement_idx:
            max_measure_idx = max(dynamic_measurement_idx)

        else:
            max_measure_idx = max(
                max(static_measurement_idx), max(dynamic_measurement_idx)
            )
        # the total rows of Q should be equal or more than the number of maximum index given by the argument
        if len(self.jacobian) < max_measure_idx * self.Nt:
            raise ValueError(
                "Inconsistent Jacobian matrix shape. Expecting at least "
                + str(max_measure_idx * self.Nt)
                + " rows in Jacobian matrix Q."
            )

        # if one measurement is in both SCM indices and DCM indices, its Jacobian is included twice (overlap is allowed)
        # compute how many measurements as SCM
        if static_measurement_idx:
            static_idx_len = len(static_measurement_idx)
        # if given None, this variable is given 0 (len(None) gives error message)
        else:
            static_idx_len = 0

        # compute how many measurements as DCM
        if dynamic_measurement_idx:
            dynamic_idx_len = len(dynamic_measurement_idx)
        # if given None, this variable is given 0 (len(None) gives error message)
        else:
            dynamic_idx_len = 0

        # compute N_total_measure, including both SCMs and DCMs. store for later use
        self.total_measure_idx = static_idx_len + dynamic_idx_len
        # all measurements have Nt time points
        total_measure_len = (static_idx_len + dynamic_idx_len) * self.Nt

        # initialize Jacobian Q as numpy array
        # here we stack the Jacobian Q according to the orders the user provides SCM and DCM index
        # jac: Jacobian matrix of shape N_total_measure * Np
        jac = np.zeros((total_measure_len, self.n_parameters))
        # update row number counter in the reassembled jac
        # starts from -1 because we add one every time before giving it a value
        update_row_counter = -1
        # if there is static-cost measurements
        if static_measurement_idx is not None:
            # loop over SCM indices
            for i in static_measurement_idx:
                # loop over time points
                for t in range(self.Nt):
                    # locate the row number in the original Jacobian information
                    row_number = i * self.Nt + t
                    # update the row number in the assembled Jacobian matrix
                    update_row_counter += 1
                    # loop over columns, i.e. parameters
                    for p in range(self.n_parameters):
                        # it maps to column p+1 in the original Jacobian, because the first column is measurement name
                        jac[update_row_counter][p] = self.jacobian[row_number][p]
        # if there is dynamic-cost measurements
        if dynamic_measurement_idx is not None:
            # loop over DCM indices
            for j in dynamic_measurement_idx:
                # loop over time points
                for t in range(self.Nt):
                    # locate the row number in the origianl Jacobian information
                    row_number = j * self.Nt + t
                    # update row number in assembled Jacobian matrix
                    update_row_counter += 1
                    # loop over columns, i.e. parameters
                    for p in range(self.n_parameters):
                        # it maps to column p+1 in the original Jacobian, because the first column is measurement name
                        jac[update_row_counter][p] = self.jacobian[row_number][p]

        self.jac = jac


@dataclass
class MeasurementData:
    """
    containing measurement related information.
    :param name: a list of strings, the measurement names
    :param jac_index: a list of int, the indices of measurements in the Jacobian matrix
    :param static_cost: a list of float, the static cost of measurements
    :param dynamic_cost: a list of float, the dynamic cost of measurements
    :param min_time_interval: float, the minimal time interval between two sampled time points
    :param max_num_sample: int, the maximum number of samples for each measurement
    :param total_max_num_sample: int, the maximum number of samples for all measurements
    """

    name: list
    jac_index: list
    static_cost: list
    dynamic_cost: list
    min_time_interval: float
    max_num_sample: int
    total_max_num_sample: int

    def _check_if_input_is_valid(self):
        """This function is to check if the input types are consistent with what this class expects.
        Adapted from: https://stackoverflow.com/questions/50563546/validating-detailed-types-in-python-dataclasses
        from the answer of @Arne
        """
        # loop over all names of this dataclass
        for field_name, field_def in self.__dataclass_fields__.items():
            # get the type the input it should be
            data_type = field_def.type
            # get the actual type we get from the user's input
            input_type = type(getattr(self, field_name))
            # check if the actual type is the same with what we expect
            if input_type != field_def.type:
                raise ValueError(
                    "Instead of getting type ", data_type, ", the input is ", input_type
                )

    def __post_init__(self):
        """This is added for error checking."""
        # if one input is not the type it is supposed to be, throw error
        self._check_if_input_is_valid()


class MeasurementOptimizer:
    def __init__(
        self,
        sens_info,
        measure_info,
        error_cov=None,
        error_opt=CovarianceStructure.identity,
        print_level=0,
    ):
        """
        Arguments
        ---------
        :param sens_info: the SensitivityData object
            containing Jacobian matrix jac and the number of timepoints Nt
            jac contains m lists, m is the number of meausrements
            Each list contains an N_t_m*n_parameters elements, which is the sensitivity matrix Q for measurement m
        :param measure_info: the MeasurementData object
            containing the string names of measurements, indices of measurements in the Jacobian matrix,
            the static costs and dynamic costs of the measurements,
            minimal interval time between two sample points, and the maximum number of samples.
        :param error_cov: a numpy array
            defined error covariance matrix here
            if CovarianceStructure.identity: error covariance matrix is an identity matrix
            if CovarianceStructure.variance: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
                Shape: Sum(Nt)
            if CovarianceStructure.time_correlation: a 3D numpy array, each element is the error covariances
                This option assumes covariances not between measurements, but between timepoints for one measurement
                Shape: Nm * (Nt_m * Nt_m)
            if CovarianceStructure.measure_correlation: a 2D numpy array, covariance matrix for a single time steps
                This option assumes the covariances between measurements at the same timestep in a time-invariant way
                Shape: Nm * Nm
            if CovarianceStructure.time_measure_correlation: a 2D numpy array, covariance matrix for the flattened measurements
                Shape: sum(Nt) * sum(Nt)
        :param: error_opt: CovarianceStructure
            can choose from identity, variance, time_correlation, measure_correlation, time_measure_correlation. See above comments.
        :param print_level: int
            indicate what statements are printed for debugging at the pre-computation stage
            0 (default): no extra printing
            1: print to show it is working
            2: print for debugging
            3: print everything that could help with debugging


        Returns
        -------
        None
        """
        # print_level received here is for the pre-computation stage
        self.precompute_print_level = print_level

        ## parse sensitivity info from SensitivityData
        # get total measurement number from the shape of Q
        self.n_total_measurements = len(sens_info.jac)

        # store the sens object
        self.sens_info = sens_info

        ## parse measurement info from MeasurementData
        self.measure_info = measure_info
        # parse measure_info information
        self._parse_measure_info()

        # check if SensitivityData and MeasurementData provide consistent inputs
        self._check_consistent_sens_measure()

        # flattened Q and indexes
        self._dynamic_flatten(sens_info.jac)

        # build and check PSD of Sigma
        # check sigma inputs
        self._check_sigma(error_cov, error_opt)

        # build the Sigma and Sigma_inv (error covariance matrix and its inverse matrix)
        Sigma = self._build_sigma(error_cov, error_opt)

        # split Sigma_inv to DCM-DCM error, DCM-SCM error vector, SCM-SCM error matrix
        self._split_sigma(Sigma)

    def _parse_measure_info(self):
        """
        This function decodes information from measure_info object.
        """
        # indices of static and dynamic measurements, stored in lists
        static_measurement_idx, dynamic_measurement_idx = [], []
        # dynamic_cost_measure_info stores the static costs of dynamic-cost measurements
        # static_cost_measure_info stores the static costs of static-cost measurements
        dynamic_cost_measure_info, static_cost_measure_info = [], []
        # loop over the number of measurements
        for i in range(len(self.measure_info.dynamic_cost)):
            # if there is no dynamic costs, this is a static-cost measurement
            if self.measure_info.dynamic_cost[i] == 0:
                # add to static-cost measurment indices list
                static_measurement_idx.append(i)
                # add its static cost to the static-cost measurements' cost list
                static_cost_measure_info.append(self.measure_info.static_cost[i])
            # if there are dynamic costs, this is a dynamic-cost measurement
            else:
                # add to dynamic-cost measurement indices list
                dynamic_measurement_idx.append(i)
                # add its static cost to the dynamic-cost measurements' cost list
                dynamic_cost_measure_info.append(self.measure_info.static_cost[i])

        if self.precompute_print_level == 3:
            print("Static-cost measurement idx:", static_measurement_idx)
            print("Dynamic-cost measurement idx:", dynamic_measurement_idx)

        # number of SCMs
        self.n_static_measurements = len(static_measurement_idx)
        # SCM indices
        self.static_measurement_idx = static_measurement_idx
        # number of DCMs
        self.n_dynamic_measurements = len(dynamic_measurement_idx)
        # DCM indices
        self.dynamic_measurement_idx = dynamic_measurement_idx
        # static-cost measurements' cost list
        self.cost_list = static_cost_measure_info
        # dynamic-cost measurements' cost list
        self.dynamic_cost_measure_info = dynamic_cost_measure_info
        # get DCM installation costs
        self.dynamic_install_cost = [
            self.measure_info.static_cost[i] for i in dynamic_measurement_idx
        ]

        # parse measurement names
        self.measure_name = self.measure_info.name  # measurement name list

        # add dynamic-cost measurements list
        # loop over DCM index list
        for i in self.dynamic_measurement_idx:
            # loop over dynamic-cost measurements time points
            for _ in range(self.sens_info.Nt):
                self.cost_list.append(self.measure_info.dynamic_cost[i])

        # total number of all measurements and all time points
        self.total_num_time = self.sens_info.Nt * (
            self.n_static_measurements + self.n_dynamic_measurements
        )

        # min time interval, only for dynamic-cost measurements
        # if there is no min time interval, it is 0
        self.min_time_interval = self.measure_info.min_time_interval

        # each manual number, for one measurement, how many time points can be chosen at most
        # if this number is >= Nt, then there is no limitation to how many of them can be chosen
        self.each_manual_number = self.measure_info.max_num_sample

        # the maximum number of time points can be chosen for all DCMs
        self.manual_number = self.measure_info.total_max_num_sample

        if self.precompute_print_level >= 2:
            print("Minimal time interval between two samples:", self.min_time_interval)
            print(
                "Maximum number of samples for each measurement:",
                self.each_manual_number,
            )
            print("Maximum number of samples for all measurements:", self.manual_number)
            if self.precompute_print_level == 3:
                # print measurement information
                print(
                    "cost list of all measurements, including SCMs and time points for DCMs:",
                    self.cost_list,
                )
                print("DCMs installation costs:", self.dynamic_cost_measure_info)

    def _check_consistent_sens_measure(self):
        """This function checks if SensitivityData and MeasurementData provide consistent inputs"""
        # check if the index list of MeasurementData is in the range of SensitivityData
        if (
            max(self.measure_info.jac_index) + 1
        ) * self.sens_info.Nt > self.n_total_measurements:
            raise ValueError(
                "The measurement index is out of the range of the given Jacobian matrix"
            )

    def _check_sigma(self, error_cov, error_option):
        """Check sigma inputs shape and values

        Arguments
        ---------
        :param error_cov: if error_cov is None, return an identity matrix
        option 1: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
            Shape: Sum(Nt)
        option 2: a 3D numpy array, each element is the error covariances
            This option assumes covariances not between measurements, but between timepoints for one measurement
            Shape: Nm * (Nt_m * Nt_m)
        option 3: a 2D numpy array, covariance matrix for a single time steps
            This option assumes the covariances between measurements at the same timestep in a time-invariant way
            Shape: Nm * Nm
        option 4: a 2D numpy array, covariance matrix for the flattened measurements
            Shape: sum(Nt) * sum(Nt)
        :param: error_opt: CovarianceStructure
            can choose from identity, variance, time_correlation, measure_correlation, time_measure_correlation. See above comments.

        Returns
        -------
        None
        """
        # identity matrix
        if (error_option == CovarianceStructure.identity) or (
            error_option == CovarianceStructure.variance
        ):
            # if None, it means identity matrix

            # if not None, need to check shape
            if error_cov is not None:
                if len(error_cov) != self.total_num_time:
                    raise ValueError(
                        "error_cov must have the same length as total_num_time. Expect length:"
                        + str(self.total_num_time)
                    )

        elif error_option == CovarianceStructure.time_correlation:
            # check the first dimension (length of DCMs)
            if len(error_cov) != self.n_total_measurements:
                raise ValueError(
                    "error_cov must have the same length as n_total_measurements. Expect length:"
                    + str(self.n_total_measurements)
                )

            # check the time correlation matrice shape for each DCM
            # loop over the index of DCM to retrieve the number of time points for DCM
            for i in range(self.n_total_measurements):
                # check row number
                if len(error_cov[0]) != self.sens_info.Nt:
                    raise ValueError(
                        "error_cov[i] must have the shape Nt*Nt. Expect number of rows:"
                        + str(self.sens_info.Nt)
                    )
                # check column number
                if len(error_cov[0][0]) != self.sens_info.Nt:
                    raise ValueError(
                        "error_cov[i] must have the shape Nt*Nt. Expect number of columns:"
                        + str(self.sens_info.Nt)
                    )

        elif error_option == CovarianceStructure.measure_correlation:
            # check row number
            if len(error_cov) != self.sens_info.total_measure_idx:
                raise ValueError(
                    "error_cov must have the same length as total_measure_idx. Expect number of rows:"
                    + str(self.sens_info.total_measure_idx)
                )
            # check column number
            if len(error_cov[0]) != self.sens_info.total_measure_idx:
                raise ValueError(
                    "error_cov[i] must have the same length as total_measure_idx. Expect number of columns:"
                    + str(self.sens_info.total_measure_idx)
                )

        elif error_option == CovarianceStructure.time_measure_correlation:
            # check row number
            if len(error_cov) != self.total_num_time:
                raise ValueError(
                    "error_cov must have the shape total_num_time*total_num_time. Expect number of rows:"
                    + str(self.total_num_time)
                )
            # check column number
            if len(error_cov[0]) != self.total_num_time:
                raise ValueError(
                    "error_cov must have the shape total_num_time*total_num_time. Expect number of columns:"
                    + str(self.total_num_time)
                )

    def _dynamic_flatten(self, jac):
        """Update dynamic flattened matrix index.
        Arguments
        ---------
        :param jac: Jacobian information for main class use, a 2D array of shape [N_total_meausrements * Np]

        Returns
        -------
        dynamic_flatten matrix: flatten dynamic-cost measurements, not flatten static-costs, [s1, d1|t1, ..., d1|tN, s2]
        Flatten matrix: flatten dynamic-cost and static-cost measuremenets
        """

        ### dynamic_flatten: to be decision matrix
        jac_dynamic_flatten = []
        # position index in jac_dynamic_flatten where each measurement starts
        self.head_pos_dynamic_flatten = {}
        # all static measurements index after dynamic_flattening
        self.static_idx_dynamic_flatten = []
        self.dynamic_idx_dynamic_flatten = []

        ### flatten: flatten all measurement all costs
        jac_flatten = []
        # position index in jac_flatten where each measurement starts
        self.head_pos_flatten = {}
        # all static measurements index after flatten
        self.static_idx_flatten = []
        # all dynamic measurements index after flatten
        self.dynamic_idx_flatten = []

        # map dynamic index to flatten index
        # key: dynamic index, value: corresponding indexes in flatten matrix. For static, it's a list. For dynamic, it's a index value
        self.dynamic_to_flatten = {}

        # counter for dynamic_flatten
        count1 = 0
        # counter for flatten
        count2 = 0
        # loop over total measurement index
        for i in range(self.sens_info.total_measure_idx):
            if (
                i in self.static_measurement_idx
            ):  # static measurements are not flattened for dynamic flatten
                if self.precompute_print_level == 3:
                    print("Static-cost measurement idx: ", i)
                # dynamic_flatten for SCM
                jacobian_static = []
                # locate head row of the sensitivity for this measurement
                head_row = i * self.sens_info.Nt
                for t in range(self.sens_info.Nt):
                    jacobian_static.append(jac[head_row + t])

                jac_dynamic_flatten.append(jacobian_static)
                # map position index in jac_dynamic_flatten where each measurement starts
                self.head_pos_dynamic_flatten[i] = count1
                # store all static measurements index after dynamic_flattening
                self.static_idx_dynamic_flatten.append(count1)
                self.dynamic_to_flatten[count1] = (
                    []
                )  # static measurement's dynamic_flatten index corresponds to a list of flattened index

                # flatten
                for t in range(self.sens_info.Nt):
                    jac_flatten.append(jac[head_row + t])
                    if t == 0:
                        self.head_pos_flatten[i] = count2
                    # all static measurements index after flatten
                    self.static_idx_flatten.append(count2)
                    # map all timepoints to the dynamic_flatten static index
                    self.dynamic_to_flatten[count1].append(count2)
                    count2 += 1

                count1 += 1

            elif i in self.dynamic_measurement_idx:
                if self.precompute_print_level == 3:
                    print("Dynamic-cost measurement idx: ", i)
                # dynamic measurements are flattend for both dynamic_flatten and flatten
                # locate head row of the sensitivity for this measurement
                head_row = i * self.sens_info.Nt
                for t in range(self.sens_info.Nt):
                    jac_dynamic_flatten.append(jac[head_row + t])
                    if t == 0:
                        self.head_pos_dynamic_flatten[i] = count1
                    self.dynamic_idx_dynamic_flatten.append(count1)

                    jac_flatten.append(jac[head_row + t])
                    if t == 0:
                        self.head_pos_flatten[i] = count2
                    self.dynamic_to_flatten[count1] = count2
                    count2 += 1

                    count1 += 1

        self.jac_dynamic_flatten = jac_dynamic_flatten
        self.jac_flatten = jac_flatten
        # dimension after dynamic_flatten
        self.num_measure_dynamic_flatten = len(self.static_idx_dynamic_flatten) + len(
            self.dynamic_idx_dynamic_flatten
        )
        # dimension after flatten
        self.num_measure_flatten = len(self.static_idx_flatten) + len(
            self.dynamic_idx_flatten
        )

        if self.precompute_print_level >= 2:
            print("Number of binary decisions:", self.num_measure_dynamic_flatten)

            if self.precompute_print_level == 3:
                print(
                    "Dimension after dynamic flatten:", self.dynamic_idx_dynamic_flatten
                )
                print("Dimension after flatten:", self.dynamic_idx_flatten)

    def _build_sigma(self, error_cov, error_option):
        """Build error covariance matrix

        Arguments
        ---------
        :param error_cov: if error_cov is None, return an identity matrix
        option 1: a numpy vector, each element is the corresponding variance, a.k.a. diagonal elements.
            Shape: Sum(Nt)
        option 2: a 3D numpy array, each element is the error covariances
            This option assumes covariances not between measurements, but between timepoints for one measurement
            Shape: Nm * (Nt_m * Nt_m)
        option 3: a 2D numpy array, covariance matrix for a single time steps
            This option assumes the covariances between measurements at the same timestep in a time-invariant way
            Shape: Nm * Nm
        option 4: a 2D numpy array, covariance matrix for the flattened measurements
            Shape: sum(Nt) * sum(Nt)
        :param: error_opt: CovarianceStructure
            can choose from identity, variance, time_correlation, measure_correlation, time_measure_correlation. See above comments.

        Returns
        -------
        Sigma: a 2D numpy array, covariance matrix for the flattened measurements
            Shape: sum(Nt) * sum(Nt)
        """

        # initialize error covariance matrix, shape N_all_t * N_all_t
        Sigma = np.zeros((self.total_num_time, self.total_num_time))

        # identity matrix or only have variance
        if (error_option == CovarianceStructure.identity) or (
            error_option == CovarianceStructure.variance
        ):
            # if given None, it means it is an identity matrix
            if error_cov is None:
                # create identity matrix
                error_cov = [1] * self.total_num_time
            # loop over diagonal elements and change
            for i in range(self.total_num_time):
                # Sigma has 0 in all off-diagonal elements, error_cov gives the diagonal elements
                Sigma[i, i] = error_cov[i]

            if self.precompute_print_level >= 2:
                print("Error covariance matrix option:", error_option)
                if self.precompute_print_level == 3:
                    print("Error matrix:", Sigma)

        # different time correlation matrix for each measurement
        # no covariance between measurements
        elif error_option == CovarianceStructure.time_correlation:
            for i in range(self.n_total_measurements):
                # give the error covariance to Sigma
                # each measurement has a different time-correlation structure
                # that is why this is a 3D matrix
                sigma_i_start = self.head_pos_flatten[i]
                # loop over all timepoints for measurement i
                # for each measurement, the time correlation matrix is Nt*Nt
                for t1 in range(self.sens_info.Nt):
                    for t2 in range(self.sens_info.Nt):
                        # for the ith measurement, the error matrix is error_cov[i]
                        Sigma[sigma_i_start + t1, sigma_i_start + t2] = error_cov[i][
                            t1
                        ][t2]

            if self.precompute_print_level >= 2:
                print("Error covariance matrix option:", error_option)
                if self.precompute_print_level == 3:
                    print("Error matrix:", Sigma)

        # covariance between measurements
        # the covariances between measurements at the same timestep in a time-invariant way
        elif error_option == CovarianceStructure.measure_correlation:
            # loop over number of measurements
            for i in range(self.sens_info.total_measure_idx):
                # loop over number of measurements
                for j in range(self.sens_info.total_measure_idx):
                    # find the covariance term
                    cov_ij = error_cov[i][j]
                    # find the starting index for each measurement (each measurement i has Nt entries)
                    head_i = self.head_pos_flatten[i]
                    # starting index for measurement j
                    head_j = self.head_pos_flatten[j]
                    # i, j may have different timesteps
                    # we find the corresponding index by locating the starting indices
                    for t in range(self.sens_info.Nt):
                        Sigma[t + head_i, t + head_j] = cov_ij

            if self.precompute_print_level >= 2:
                print("Error covariance matrix option:", error_option)
                if self.precompute_print_level == 3:
                    print("Error matrix:", Sigma)

        # the full covariance matrix is given
        elif error_option == CovarianceStructure.time_measure_correlation:
            Sigma = np.asarray(error_cov)

            if self.precompute_print_level >= 2:
                print("Error covariance matrix option:", error_option)
                if self.precompute_print_level == 3:
                    print("Error matrix:", Sigma)

        self.Sigma = Sigma

        return Sigma

    def _split_sigma(self, Sigma):
        """Split the error covariance matrix to be used for computation
        They are split to DCM-DCM (scalar) covariance, DCM-SCM (vector) covariance, SCCM-SCM (matrix) covariance
        We inverse the Sigma for the computation of FIM

        Arguments
        ---------
        :param Sigma: a 2D numpy array, covariance matrix for the flattened measurements
            Shape: sum(Nt) * sum(Nt)

        Returns
        -------
        None
        """
        # Inverse of covariance matrix is used
        # pinv is used to avoid ill-conditioning issues
        Sigma_inv = np.linalg.pinv(Sigma)
        self.Sigma_inv_matrix = Sigma_inv
        # Use a dicionary to store the inverse of sigma as either scalar number, vector, or matrix
        self.Sigma_inv = {}

        # between static and static: (Nt_i+Nt_j)*(Nt_i+Nt_j) matrix
        for i in self.static_idx_dynamic_flatten:  # loop over static measurement index
            for (
                j
            ) in self.static_idx_dynamic_flatten:  # loop over static measurement index
                # should be a (Nt_i+Nt_j)*(Nt_i+Nt_j) matrix
                sig = np.zeros((self.sens_info.Nt, self.sens_info.Nt))
                # row [i, i+Nt_i], column [i, i+Nt_i]
                for ti in range(self.sens_info.Nt):  # loop over time points
                    for tj in range(self.sens_info.Nt):  # loop over time points
                        sig[ti, tj] = Sigma_inv[
                            self.head_pos_flatten[i] + ti, self.head_pos_flatten[j] + tj
                        ]
                self.Sigma_inv[(i, j)] = sig

        # between static and dynamic: Nt*1 matrix
        for i in self.static_idx_dynamic_flatten:  # loop over static measurement index
            for (
                j
            ) in (
                self.dynamic_idx_dynamic_flatten
            ):  # loop over dynamic measuremente index
                # should be a vector, here as a Nt*1 matrix
                sig = np.zeros((self.sens_info.Nt, 1))
                # row [i, i+Nt_i], col [j]
                for t in range(self.sens_info.Nt):  # loop over time points
                    # print(i,j)
                    # print(t, self.head_pos_flatten[i], self.dynamic_to_flatten[j])
                    sig[t, 0] = Sigma_inv[
                        self.head_pos_flatten[i] + t, self.dynamic_to_flatten[j]
                    ]
                self.Sigma_inv[(i, j)] = sig

        # between static and dynamic: Nt*1 matrix
        for (
            i
        ) in self.dynamic_idx_dynamic_flatten:  # loop over dynamic measurement index
            for (
                j
            ) in self.static_idx_dynamic_flatten:  # loop over static measurement index
                # should be a vector, here as Nt*1 matrix
                sig = np.zeros((self.sens_info.Nt, 1))
                # row [j, j+Nt_j], col [i]
                for t in range(self.sens_info.Nt):  # loop over time
                    sig[t, 0] = Sigma_inv[
                        self.head_pos_flatten[j] + t, self.dynamic_to_flatten[i]
                    ]
                self.Sigma_inv[(i, j)] = sig

        # between dynamic and dynamic: a scalar number
        for (
            i
        ) in self.dynamic_idx_dynamic_flatten:  # loop over dynamic measurement index
            for (
                j
            ) in (
                self.dynamic_idx_dynamic_flatten
            ):  # loop over dynamic measurement index
                # should be a scalar number
                self.Sigma_inv[(i, j)] = Sigma_inv[
                    self.dynamic_to_flatten[i], self.dynamic_to_flatten[j]
                ]

    def assemble_unit_fims(self):
        """
        compute a list of FIM.
        unit FIMs include DCM-DCM FIM, DCM-SCM FIM, SCM-SCM FIM
        """

        self.unit_fims = []

        # loop over measurement index
        for i in range(self.num_measure_dynamic_flatten):
            # loop over measurement index
            for j in range(self.num_measure_dynamic_flatten):

                # static*static
                if (
                    i in self.static_idx_dynamic_flatten
                    and j in self.static_idx_dynamic_flatten
                ):
                    # print("static * static, cov:", self.Sigma_inv[(i,j)])
                    unit = (
                        np.asarray(self.jac_dynamic_flatten[i]).T
                        @ self.Sigma_inv[(i, j)]
                        @ np.asarray(self.jac_dynamic_flatten[j])
                    )

                # consider both i=SCM, j=DCM scenario and i=DCM, j=SCM scenario
                # static*dynamic
                elif (
                    i in self.static_idx_dynamic_flatten
                    and j in self.dynamic_idx_dynamic_flatten
                ):
                    # print("static*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = (
                        np.asarray(self.jac_dynamic_flatten[i]).T
                        @ self.Sigma_inv[(i, j)]
                        @ np.asarray(self.jac_dynamic_flatten[j]).reshape(
                            1, self.sens_info.n_parameters
                        )
                    )

                # static*dynamic
                elif (
                    i in self.dynamic_idx_dynamic_flatten
                    and j in self.static_idx_dynamic_flatten
                ):
                    # print("static*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = (
                        np.asarray(self.jac_dynamic_flatten[i])
                        .reshape(1, self.sens_info.n_parameters)
                        .T
                        @ self.Sigma_inv[(i, j)].T
                        @ np.asarray(self.jac_dynamic_flatten[j])
                    )

                # dynamic*dynamic
                else:
                    # print("dynamic*dynamic, cov:", self.Sigma_inv[(i,j)])
                    unit = (
                        self.Sigma_inv[(i, j)]
                        * np.asarray(self.jac_dynamic_flatten[i])
                        .reshape(1, self.sens_info.n_parameters)
                        .T
                        @ np.asarray(self.jac_dynamic_flatten[j]).reshape(
                            1, self.sens_info.n_parameters
                        )
                    )

                # check if unitFIM is symmetric
                # Note: I removed this function because unit FIM doesn't need to be symmetric
                # f not np.allclose(unit, unit.T):
                #   if self.precompute_print_level == 3:
                #       print("Index ",i,j, "has not symmetric FIM:", unit)
                #   raise ValueError("The unit FIM is not symmetric with index:", i, j)

                # store unit FIM following this order
                self.unit_fims.append(unit.tolist())

        if self.precompute_print_level >= 1:
            print("Number of unit FIMs:", len(self.unit_fims))

    def _measure_matrix(self, measurement_vector):
        """
        This is a helper function, when giving a vector of solutions, it converts this vector into a 2D array
        This is needed for validating the solutions after the optimization,
        since we only computes the half diagonal of the measurement matrice and flatten it.

        Arguments
        ---------
        :param measurement_vector: a vector of measurement weights solution

        Returns
        -------
        measurement_matrix: a full measurement matrix, construct the weights for covariances
        """
        # check if measurement vector legal
        if len(measurement_vector) != self.total_no_measure:
            raise ValueError(
                "Measurement vector is of wrong shape, expecting length of"
                + str(self.total_no_measure)
            )

        # initialize measurement matrix as a 2D array
        measurement_matrix = np.zeros((self.total_no_measure, self.total_no_measure))

        # loop over total measurement index
        for i in range(self.total_no_measure):
            for j in range(self.total_no_measure):
                measurement_matrix[i, j] = min(
                    measurement_vector[i], measurement_vector[j]
                )

        return measurement_matrix

    def _continuous_optimization(
        self,
        mixed_integer=False,
        fixed_nlp=False,
        fix=False,
        upper_diagonal_only=False,
        num_dynamic_t_name=None,
        budget=100,
        init_cov_y=None,
        initial_fim=None,
        dynamic_install_initial=None,
        total_measure_initial=1,
        static_dynamic_pair=None,
        time_interval_all_dynamic=False,
        multi_component_pairs=None,
        max_num_z=None,
        max_num_o=None,
        max_num_z_lists=None,
        max_num_o_lists=None,
        max_lists=None,
        total_manual_num_init=10,
        cost_initial=100,
        fim_diagonal_small_element=0,
        print_level=0,
    ):
        """Continuous optimization problem formulation.

        Arguments
        ---------
        :param mixed_integer: boolean
            not relaxing integer decisions
        :param fixed_nlp: boolean
            if True, the problem is formulated as a fixed NLP
        :param fix: boolean
            if solving as a square problem or with DOFs
        :param upper_diagonal_only: boolean
            if using upper_diagonal_only set to define decisions and FIM, or not
        :param num_dynamic_t_name: list
            a list of the exact time points for the dynamic-cost measurements time points
        :param budget: integer
            total budget
        :param init_cov_y: list of lists
            initialize decision variables
        :param initial_fim: list of lists
            initialize FIM
        :param dynamic_install_initial: list
            initialize if_dynamic_install
        :param total_measure_initial: integer
            initialize the total number of measurements chosen
        :param static_dynamic_pair: list of lists
            a list of the name of measurements, that are selected as either dynamic or static measurements.
        :param time_interval_all_dynamic: boolean
            if True, the minimal time interval applies for all dynamical measurements
        :param total_manual_num_init: integer
            initialize the total number of dynamical timepoints selected
        :param cost initial: float
            initialize the cost
        :param fim_diagonal_small_element: float
            a small number, default to be 0, to be added to FIM diagonal elements for better convergence
        :param print_level: integer
            0 (default): no extra printing
            1: print to show it is working
            2: print for debugging
            3: print everything that could help with debugging

        Returns
        -------
        None
        """

        m = pyo.ConcreteModel()

        # measurements set
        m.n_responses = pyo.Set(initialize=range(self.num_measure_dynamic_flatten))
        # FIM set
        m.dim_fim = pyo.Set(initialize=range(self.sens_info.n_parameters))

        # set up pyomo parameters
        m.budget = pyo.Param(initialize=budget, mutable=True)
        m.fim_diagonal_small_element = pyo.Param(
            initialize=fim_diagonal_small_element, mutable=True
        )
        # this element needs to be stored in object for warmstart function use
        # because warmstart function won't work for pyomo model component
        self.fim_diagonal_small_element = fim_diagonal_small_element

        # print_level received here is for the optimization algorithm debugging
        self.optimize_print_level = print_level

        self.fixed_nlp = fixed_nlp
        self.initial_fim = initial_fim
        # dynamic measurements parameters
        # dynamic measurement number of timepoints
        self.dynamic_Nt = self.sens_info.Nt
        # dynamic measurement index number
        # Pyomo model explicitly numbers all of the static measurements first and then all of the dynmaic measurements
        m.dim_dynamic = pyo.Set(
            initialize=range(
                self.n_static_measurements, self.sens_info.total_measure_idx
            )
        )
        # turn dynamic measurement number of timepoints into a pyomo set
        m.dim_dynamic_t = pyo.Set(initialize=range(self.dynamic_Nt))

        # pair time index and real time
        # for e.g., time 2h is the index 16, dynamic[16] = 2
        # this is for the time interval between two DCMs computation
        if num_dynamic_t_name is not None:
            dynamic_time = {}
            # loop over time index
            for i in range(self.dynamic_Nt):
                # index: real time
                dynamic_time[i] = num_dynamic_t_name[i]
            self.dynamic_time = dynamic_time

        # initialize with identity
        def identity(m, a, b):
            return 1 if a == b else 0

        def initialize_point(m, a, b):
            if init_cov_y[a][b] > 0:
                return init_cov_y[a][b]
            else:
                # this is to avoid that some times the given solution contains a really small negative number
                return 0

        if init_cov_y is not None:
            initialize = initialize_point
        else:
            initialize = identity

        if self.optimize_print_level == 3:
            print("Binary solution matrix is initialized with:", initialize)

        # only define the upper triangle of symmetric matrices
        def n_responses_half_init(m):
            return (
                (a, b)
                for a in m.n_responses
                for b in range(a, self.num_measure_dynamic_flatten)
            )

        # only define the upper triangle of FIM
        def dim_fim_half_init(m):
            return (
                (a, b) for a in m.dim_fim for b in range(a, self.sens_info.n_parameters)
            )

        # set for measurement y matrix
        m.responses_upper_diagonal = pyo.Set(dimen=2, initialize=n_responses_half_init)
        # set for FIM
        m.dim_fim_half = pyo.Set(dimen=2, initialize=dim_fim_half_init)

        # if decision variables y are binary
        if mixed_integer:
            y_domain = pyo.Binary
        # if decision variables y are relaxed
        else:
            y_domain = pyo.NonNegativeReals

        # if only defining upper triangle of the y matrix
        if upper_diagonal_only:
            m.cov_y = pyo.Var(
                m.responses_upper_diagonal,
                initialize=initialize,
                bounds=(0, 1),
                within=y_domain,
            )
        # else, define all elements in the y symmetry matrix
        else:
            m.cov_y = pyo.Var(
                m.n_responses,
                m.n_responses,
                initialize=initialize,
                bounds=(0, 1),
                within=y_domain,
            )

        # use a fix option to compute results for square problems with given y
        if fix or fixed_nlp:
            m.cov_y.fix()

        def init_fim(m, p, q):
            """initialize FIM"""
            if not initial_fim:  # if None, use identity matrix
                if p == q:
                    return 1
                else:
                    return 0

            return initial_fim[p, q]

        if self.optimize_print_level >= 2:
            print("FIM is initialized with:", initial_fim)

        if upper_diagonal_only:
            m.total_fim = pyo.Var(m.dim_fim_half, initialize=init_fim)
        else:
            m.total_fim = pyo.Var(m.dim_fim, m.dim_fim, initialize=init_fim)

        ### compute FIM
        def eval_fim(m, a, b):
            """
            Add constraints to calculate FIM from unit contributions
            FIM = sum(cov_y[i,j]*unit FIM[i,j]) for all i, j in n_responses

            a, b: indices for FIM, iterate in parameter set
            """
            if a <= b:
                summi = 0
                for i in m.n_responses:
                    for j in m.n_responses:
                        # large_idx, small_idx are needed because cov_y is only defined the upper triangle matrix
                        # the FIM order is i*num_measurement + j no matter if i is the smaller one or the bigger one
                        large_idx = max(i, j)
                        small_idx = min(i, j)
                        summi += (
                            m.cov_y[small_idx, large_idx]
                            * self.unit_fims[i * self.num_measure_dynamic_flatten + j][
                                a
                            ][b]
                        )

                # if diagonal elements, a small element can be added to avoid rank deficiency
                if a == b:
                    return m.total_fim[a, b] == summi + m.fim_diagonal_small_element
                # if not diagonal, no need to add small number
                else:
                    return m.total_fim[a, b] == summi
            # FIM is symmetric so no need to compute again
            else:
                return m.total_fim[a, b] == m.total_fim[b, a]

        def integer_cut_0(m):
            """Compute the total number of measurements and time points selected
            This is for the inequality constraint of integer cut.
            """
            return m.total_number_measurements == sum(
                m.cov_y[i, i] for i in range(self.num_measure_dynamic_flatten)
            )

        def integer_cut_0_ineq(m):
            """Ensure that at least one measurement or time point is selected
            integer cut that cuts the solution of all 0
            """
            return m.total_number_measurements >= 1

        def total_dynamic(m):
            """compute the total number of time points from DCMs are selected
            This is for the inequality constraint of total number of time points from DCMs < total number of measurements limit
            """
            return m.total_number_dynamic_measurements == sum(
                m.cov_y[i, i]
                for i in range(
                    self.n_static_measurements, self.num_measure_dynamic_flatten
                )
            )

        ### cov_y constraints
        def y_covy1(m, a, b):
            """
            cov_y[a,b] indicates if measurement a, b are both selected, i.e. a & b
            cov_y[a,b] = cov_y[a,a]*cov_y[b,b]. Relax this equation to get cov_y[a,b] <= cov_y[a,a]
            """
            if b > a:
                return m.cov_y[a, b] <= m.cov_y[a, a]
            else:
                # skip lower triangle constraints since y is a symmetric matrix
                return pyo.Constraint.Skip

        def y_covy2(m, a, b):
            """
            cov_y[a,b] indicates if measurement a, b are both selected, i.e. a & b
            cov_y[a,b] = cov_y[a,a]*cov_y[b,b]. Relax this equation to get cov_y[a,b] <= cov_y[b,b]
            """
            if b > a:
                return m.cov_y[a, b] <= m.cov_y[b, b]
            else:
                # skip lower triangle constraints since y is a symmetric matrix
                return pyo.Constraint.Skip

        def y_covy3(m, a, b):
            """
            cov_y[a,b] indicates if measurement a, b are both selected, i.e. a & b
            cov_y[a,b] = cov_y[a,a]*cov_y[b,b]. Relax this equation to get cov_y[a,b] >= cov_y[a,a]+cov_y[b,b]-1
            """
            if b > a:
                return m.cov_y[a, b] >= m.cov_y[a, a] + m.cov_y[b, b] - 1
            else:
                # skip lower triangle constraints since y is a symmetric matrix
                return pyo.Constraint.Skip

        def symmetry(m, a, b):
            """
            Ensure the symmetry of y matrix.
            This function is only called if upper_diagonal_only = False.
            This is only modeled if we model the whole FIM, instead of defining only upper triangle of y
            """
            if a < b:
                return m.cov_y[a, b] == m.cov_y[b, a]
            else:
                # skip lower triangle constraints since y is a symmetric matrix
                return pyo.Constraint.Skip

        ### cost constraints
        def cost_compute(m):
            """Compute cost
            cost = static-cost measurement cost + dynamic-cost measurement installation cost + dynamic-cost meausrement timepoint cost
            """
            static_and_dynamic_cost = sum(
                m.cov_y[i, i] * self.cost_list[i] for i in m.n_responses
            )
            dynamic_fixed_cost = sum(
                m.if_install_dynamic[j]
                * self.dynamic_install_cost[j - self.n_static_measurements]
                for j in m.dim_dynamic
            )
            return m.cost == static_and_dynamic_cost + dynamic_fixed_cost

        def cost_limit(m):
            """Total cost smaller than the given budget"""
            return m.cost <= m.budget

        def total_dynamic_con(m):
            """total number of manual dynamical measurements number"""
            return m.total_number_dynamic_measurements <= self.manual_number

        def dynamic_fix_yd(m, i, j):
            """if the install cost of one dynamical measurements should be considered
            If no timepoints are chosen, there is no need to include this installation cost
            """
            # map measurement index i to its dynamic_flatten index
            start = (
                self.n_static_measurements
                + (i - self.n_static_measurements) * self.dynamic_Nt
                + j
            )
            return m.if_install_dynamic[i] >= m.cov_y[start, start]

        def dynamic_fix_yd_con2(m, i):
            """if the install cost of one dynamical measurements should be considered"""
            # start index is the first time point idx for this measurement
            start = (
                self.n_static_measurements
                + (i - self.n_static_measurements) * self.dynamic_Nt
            )
            # end index is the last time point idx for this measurement
            end = (
                self.n_static_measurements
                + (i - self.n_static_measurements + 1) * self.dynamic_Nt
            )
            # if no any time points from this DCM is selected, its installation cost should not be included
            return m.if_install_dynamic[i] <= sum(
                m.cov_y[j, j] for j in range(start, end)
            )

        # add constraints depending on if the FIM is defined as half triangle or all
        if upper_diagonal_only:
            m.total_fim_constraint = pyo.Constraint(m.dim_fim_half, rule=eval_fim)
        else:
            m.total_fim_constraint = pyo.Constraint(m.dim_fim, m.dim_fim, rule=eval_fim)
            # only when the mixed-integer problem y are defined not only upper triangle
            # this can help the performance of MIP problems so we keep it although not use it now
            if mixed_integer:
                m.sym = pyo.Constraint(m.n_responses, m.n_responses, rule=symmetry)

        # if given fixed solution, no need to add the following constraints
        if not fix:
            # total dynamic timepoints number
            m.total_number_dynamic_measurements = pyo.Var(
                initialize=total_manual_num_init
            )
            # compute total dynamic timepoints number
            m.manual = pyo.Constraint(rule=total_dynamic)
            # total dynamic timepoints < total manual number
            m.con_manual = pyo.Constraint(rule=total_dynamic_con)

            ## integer cuts
            # intiialize total number of measurements selected
            m.total_number_measurements = pyo.Var(initialize=total_measure_initial)
            # compute total number of measurements selected
            m.integer_cut0 = pyo.Constraint(rule=integer_cut_0)
            # let total number of measurements selected > 0, so we cut the all 0 solution
            m.integer_cut0_in = pyo.Constraint(rule=integer_cut_0_ineq)

            # relaxation constraints for y[a,b] = y[a]*y[b]
            m.cov1 = pyo.Constraint(m.n_responses, m.n_responses, rule=y_covy1)
            m.cov2 = pyo.Constraint(m.n_responses, m.n_responses, rule=y_covy2)
            m.cov3 = pyo.Constraint(m.n_responses, m.n_responses, rule=y_covy3)

            # dynamic-cost measurements installaction cost
            def dynamic_install_init(m, j):
                # if there is no installation cost
                if dynamic_install_initial is None:
                    return 0
                # if there is installation cost
                else:
                    return dynamic_install_initial[j - self.n_static_measurements]

            # we choose that if this is a mixed-integer problem, the dynamic installation flag is in {0,1}
            if mixed_integer:
                var_domain = pyo.Binary
            else:
                var_domain = pyo.NonNegativeReals
            m.if_install_dynamic = pyo.Var(
                m.dim_dynamic,
                initialize=dynamic_install_init,
                bounds=(0, 1),
                within=var_domain,
            )

            # for solving fixed problem, we fix the dynamic installation flag
            if self.fixed_nlp:
                m.if_install_dynamic.fix()

            m.dynamic_cost = pyo.Constraint(
                m.dim_dynamic, m.dim_dynamic_t, rule=dynamic_fix_yd
            )
            m.dynamic_con2 = pyo.Constraint(m.dim_dynamic, rule=dynamic_fix_yd_con2)

            # if multi_component_pairs is not None:

            # for a, b in multi_component_pairs:
            #    def multi_comp_rules(m):
            #        return m.cov_y[a,a] == m.cov_y[b,b]
            #    con_name = "multi_comp_con_"+str(a)+"_"+str(b)
            #    m.add_component(con_name, pyo.Constraint(expr=multi_comp_rules))

            # if max_num_z is not None:
            for k, list_range in enumerate(max_lists):

                def max_num_each_rule(m):
                    chosen_num = sum(m.cov_y[i, i] for i in list_range)
                    return chosen_num <= max_num_z * max_num_o

                con_name = "max_loc_con_" + str(k)
                m.add_component(con_name, pyo.Constraint(expr=max_num_each_rule))

            for k, list_range in enumerate(max_num_z_lists):

                def max_num_z_rule(m):
                    chosen_num = sum(m.cov_y[i, i] for i in list_range)
                    return chosen_num <= max_num_z

                con_name = "max_z_con_" + str(k)
                m.add_component(con_name, pyo.Constraint(expr=max_num_z_rule))

            # if max_num_o is not None:

            for k, list_range in enumerate(max_num_o_lists):

                def max_num_o_rule(m):
                    chosen_num = sum(m.cov_y[i, i] for i in list_range)
                    return chosen_num <= max_num_o

                con_name = "max_o_con_" + str(k)
                m.add_component(con_name, pyo.Constraint(expr=max_num_o_rule))

            # if each manual number smaller than a given limit
            if self.each_manual_number is not None:
                # loop over dynamical measurements
                for i in range(self.n_dynamic_measurements):

                    def dynamic_manual_num(m):
                        """the timepoints for each dynamical measurement should be smaller than a given limit"""
                        start = (
                            self.n_static_measurements + i * self.dynamic_Nt
                        )  # the start index of this dynamical measurement
                        end = (
                            self.n_static_measurements + (i + 1) * self.dynamic_Nt
                        )  # the end index of this dynamical measurement
                        cost = sum(m.cov_y[j, j] for j in range(start, end))
                        return cost <= self.each_manual_number

                    con_name = "con" + str(i)
                    m.add_component(con_name, pyo.Constraint(expr=dynamic_manual_num))

            # if some measurements can only be dynamic or static
            if static_dynamic_pair is not None:
                # loop over the index of the static, and dynamic measurements
                for i, pair in enumerate(static_dynamic_pair):

                    def static_dynamic_pair_con(m):
                        return (
                            m.if_install_dynamic[pair[1]] + m.cov_y[pair[0], pair[0]]
                            <= 1
                        )

                    con_name = "con_sta_dyn" + str(i)
                    m.add_component(
                        con_name, pyo.Constraint(expr=static_dynamic_pair_con)
                    )

            # if there is minimal interval constraint
            if self.min_time_interval is not None:
                # if this constraint applies to all dynamic measurements
                if time_interval_all_dynamic:
                    for t in range(self.dynamic_Nt):
                        # end time is an open end of the region, so another constraint needs to be added to include end_time
                        # if dynamic_time[t]+discretize_time <= end_time+0.1*discretize_time:
                        def discretizer(m):
                            sumi = 0
                            count = 0
                            # get the timepoints in this interval
                            while (count + t < self.dynamic_Nt) and (
                                dynamic_time[count + t] - dynamic_time[t]
                            ) < self.min_time_interval:
                                for i in m.dim_dynamic:
                                    surro_idx = (
                                        self.n_static_measurements
                                        + (i - self.n_static_measurements)
                                        * self.dynamic_Nt
                                        + t
                                        + count
                                    )
                                    sumi += m.cov_y[surro_idx, surro_idx]
                                count += 1

                            return sumi <= 1

                        con_name = "con_discreti_" + str(i) + str(t)
                        m.add_component(con_name, pyo.Constraint(expr=discretizer))
                # if this constraint applies to each dynamic measurements, in a local way
                else:
                    for i in m.dim_dynamic:
                        for t in range(self.dynamic_Nt):
                            # end time is an open end of the region, so another constraint needs to be added to include end_time
                            # if dynamic_time[t]+discretize_time <= end_time+0.1*discretize_time:

                            def discretizer(m):
                                # sumi is the summation of all measurements selected during this time interval
                                sumi = 0
                                # count helps us go through each time points in this time interval
                                count = 0
                                # get timepoints in this interval
                                while (count + t < self.dynamic_Nt) and (
                                    dynamic_time[count + t] - dynamic_time[t]
                                ) < self.min_time_interval:
                                    # surro_idx gets the index of the current time point
                                    surro_idx = (
                                        self.n_static_measurements
                                        + (i - self.n_static_measurements)
                                        * self.dynamic_Nt
                                        + t
                                        + count
                                    )
                                    # sum up all timepoints selections
                                    sumi += m.cov_y[surro_idx, surro_idx]
                                    count += 1

                                return sumi <= 1

                            con_name = "con_discreti_" + str(i) + str(t)
                            m.add_component(con_name, pyo.Constraint(expr=discretizer))

            # total cost
            m.cost = pyo.Var(initialize=cost_initial)
            # compute total cost
            m.cost_compute = pyo.Constraint(rule=cost_compute)
            # make total cost < budget
            m.budget_limit = pyo.Constraint(rule=cost_limit)

            # add model to object. This model lacks objective function which will be added separately later
            self.mod = m

    def _add_objective(self, obj=ObjectiveLib.A, mix_obj=False, alpha=1):
        """
        Add objective function to the model.

        This function is built for a consideration to easily add other objective functions
        since this is a comparatively large code module.

        Arguments
        ---------
        :param obj: Enum
            "A" or "D" optimality, use trace or determinant of FIM
        :param mix_obj: boolean
            if True, the objective function is a weighted sum of A- and D-optimality (trace and determinant)
        :param alpha: float
            range [0,1], weight of mix_obj. if 1, it is A-optimality. if 0, it is D-optimality
        """

        # set up design criterion
        def compute_trace(m):
            """compute trace
            trace = sum(diag(M))
            """
            sum_x = sum(m.total_fim[j, j] for j in m.dim_fim)
            return sum_x

        # set objective
        if obj == ObjectiveLib.A:  # A-optimailty
            self.mod.Obj = pyo.Objective(rule=compute_trace, sense=pyo.maximize)

        elif obj == ObjectiveLib.D:  # D-optimality

            def _model_i(b):
                # build grey-box module
                self._build_model_external(b)

            self.mod.my_block = pyo.Block(rule=_model_i)

            # loop over parameters
            for i in range(self.sens_info.n_parameters):
                # loop over upper triangle of FIM
                for j in range(i, self.sens_info.n_parameters):

                    def eq_fim(m):
                        """Make FIM in this model equal to the FIM computed by grey-box. Necessary."""
                        return m.total_fim[i, j] == m.my_block.egb.inputs[(i, j)]

                    con_name = "con" + str(i) + str(j)
                    self.mod.add_component(con_name, pyo.Constraint(expr=eq_fim))

            # add objective
            # if mix_obj, we use a weighted sum of A- and D-optimality
            if mix_obj:
                self.mod.trace = pyo.Expression(rule=compute_trace)
                self.mod.logdet = pyo.Expression(rule=m.my_block.egb.outputs["log_det"])
                # obj is a weighted sum, alpha in [0,1] is the weight of A-optimality
                # when alpha=0, it mathematically equals to D-opt, when alpha=1, A-opt
                self.mod.Obj = pyo.Objective(
                    expr=self.mod.logdet + alpha * self.mod.trace, sense=pyo.maximize
                )

            else:
                # maximize logdet obj
                self.mod.Obj = pyo.Objective(
                    expr=self.mod.my_block.egb.outputs["log_det"], sense=pyo.maximize
                )

    def _build_model_external(self, m, fim_init=None):
        """Build the model through grey-box module

        Arguments
        ---------
        :param m: a pyomo model
        :param fim_init: an array to initialize the FIM value in grey-box model

        Return
        ------
        None
        """
        # use the same print_level as the pyomo model
        ex_model = LogDetModel(
            n_parameters=self.sens_info.n_parameters,
            initial_fim=fim_init,
            print_level=self.optimize_print_level,
        )
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model)

    def compute_fim(self, measurement_vector):
        """
        Compute a total FIM given a set of measurement choice solutions
        This is a helper function to verify solutions; It is not involved in the optimization part.
        Each unit FIM is computed as:
        FIM = Q1.T@y@Sigma_inv@Q2, Q is Jacobian

        Arguments
        ---------
        :param measurement_vector: a list of the length of all measurements, each element in [0,1]
            0 indicates this measurement is not selected, 1 indicates selected
            Note: Ensure the order of this list is the same as the order of Q, i.e. [CA(t1), ..., CA(tN), CB(t1), ...]

        Return
        ------
        FIM: a numpy array containing the total FIM given this solution
        """
        # generate measurement matrix
        measurement_matrix = self._measure_matrix(measurement_vector)

        # compute FIM as Np*Np
        fim = np.zeros((self.no_param, self.no_param))

        # go over all measurement index
        for m1_rank in range(self.total_no_measure):
            # get the corresponding gradient vector for this measurement
            # use np.matrix to keep the dimensions for the vector, otherwise numpy makes the vector 1D instead of Np*1
            jac_m1 = np.matrix(self.jac[m1_rank])
            # go over all measurement index
            for m2_rank in range(self.total_no_measure):
                # get the corresponding gradient vector for this measurement
                jac_m2 = np.matrix(self.jac[m2_rank])
                # solution of if these two measurements are selected
                measure_matrix = np.matrix([measurement_matrix[m1_rank, m2_rank]])
                # retrieve the error covariance matrix corresponding part
                sigma = np.matrix([self.Sigma_inv[m1_rank, m2_rank]])
                # compute FIM as Q.T@y@error_cov@Q
                fim_unit = jac_m1.T @ measure_matrix @ sigma @ jac_m2
                fim += fim_unit
        # FIM read
        if self.verbose:
            print_fim(fim)

        return fim

    def solve(
        self, mip_option=False, objective=ObjectiveLib.A, degeneracy_hunter=False
    ):
        """
        Set up solvers, solve the problem, and check the solver status and termination conditions

        Arguments
        ---------
        :mip_option: boolean, if True, it is a mixed-integer problem, otherwise it is a relaxed problem with no integer decisions
        :objective: Enum, "A" or "D" optimality, use trace or determinant of FIM
        :degeneracy_hunter: boolean, when set up to True, use degeneracy hunter to check infeasibility in constraints. For debugging.

        Return
        ------
        None
        """
        if self.fixed_nlp:
            solver = pyo.SolverFactory("cyipopt")
            solver.config.options["hessian_approximation"] = "limited-memory"
            additional_options = {
                "max_iter": 3000,
                "output_file": "console_output",
                "linear_solver": "mumps",
                # "halt_on_ampl_error": "yes", # this option seems not working for cyipopt
                "bound_push": 1e-10,
            }

            if degeneracy_hunter:
                additional_options = {
                    "max_iter": 0,
                    "output_file": "console_output",
                    "linear_solver": "mumps",
                    "bound_push": 1e-10,
                }

            # copy solver options
            for k, v in additional_options.items():
                solver.config.options[k] = v
            results = solver.solve(self.mod, tee=True)

        elif not mip_option and objective == ObjectiveLib.A:
            solver = pyo.SolverFactory('ipopt')
            # solver.options['linear_solver'] = "ma57"
            results = solver.solve(self.mod, tee=True)

            # solver = pyo.SolverFactory("gurobi", solver_io="python")
            # solver.options["mipgap"] = 0.1
            # results = solver.solve(self.mod, tee=True)

        elif mip_option and objective == ObjectiveLib.A:
            solver = pyo.SolverFactory("gurobi", solver_io="python")
            # solver.options['mipgap'] = 0.1
            results = solver.solve(self.mod, tee=True)

        elif not mip_option and objective == ObjectiveLib.D:
            solver = pyo.SolverFactory("cyipopt")
            solver.config.options["hessian_approximation"] = "limited-memory"
            additional_options = {
                "max_iter": 3000,
                "output_file": "console_output",
                "linear_solver": "mumps",
            }

            if degeneracy_hunter:
                additional_options = {
                    "max_iter": 0,
                    "output_file": "console_output",
                    "linear_solver": "mumps",
                    "bound_push": 1e-6,
                }

            # copy solver options
            for k, v in additional_options.items():
                solver.config.options[k] = v
            results = solver.solve(self.mod, tee=True)

        elif mip_option and objective == ObjectiveLib.D:

            solver = pyo.SolverFactory("mindtpy")

            results = solver.solve(
                self.mod,
                strategy="OA",
                init_strategy="rNLP",
                # init_strategy='initial_binary',
                mip_solver="gurobi",
                nlp_solver="cyipopt",
                calculate_dual_at_solution=True,
                tee=True,
                # call_before_subproblem_solve=self.customized_warmstart,
                # add_no_good_cuts=True,
                stalling_limit=1000,
                iteration_limit=150,
                mip_solver_tee=True,
                mip_solver_args={"options": {"NumericFocus": "3"}},
                nlp_solver_tee=True,
                nlp_solver_args={
                    "options": {
                        "hessian_approximation": "limited-memory",
                        "output_file": "console_output",
                        # "linear_solver": "mumps",
                        "max_iter": 3000,
                        # "halt_on_ampl_error": "yes",
                        "bound_push": 1e-10,
                        "warm_start_init_point": "yes",
                        "warm_start_bound_push": 1e-10,
                        "warm_start_bound_frac": 1e-10,
                        "warm_start_slack_bound_frac": 1e-10,
                        "warm_start_slack_bound_push": 1e-10,
                        "warm_start_mult_bound_push": 1e-10,
                    }
                },
            )

        # check solver status and report
        # if convergted to optimal solution and feasible
        if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal
        ):
            if self.optimize_print_level >= 2:  # print level is higher
                print("The problem is solved optimal and feasible. ")
        # if get infeasible status
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            if self.optimize_print_level >= 1:  # print level is lower to warn
                print("The problem is solved to infeasible point. ")
        # if there is other solver status
        else:
            if self.optimize_print_level >= 1:
                print("Not converged. Solver status:", results.solver.status)

    def optimizer(
        self,
        budget_opt,
        initial_solution,
        mixed_integer=False,
        obj=ObjectiveLib.A,
        mix_obj=False,
        alpha=1,
        fixed_nlp=False,
        fix=False,
        upper_diagonal_only=False,
        num_dynamic_t_name=None,
        init_cov_y=None,
        initial_fim=None,
        dynamic_install_initial=None,
        total_measure_initial=1,
        static_dynamic_pair=None,
        time_interval_all_dynamic=False,
        multi_component_pairs=None,
        max_num_z=None,
        max_num_o=None,
        max_num_z_lists=None,
        max_num_o_lists=None,
        max_lists=None,
        total_manual_num_init=10,
        cost_initial=100,
        fim_diagonal_small_element=0,
        print_level=0,
    ):
        """
        Initialize, formulate, and solve the MO problem.
        This function includes two steps:
        1) Create the optimization problem (Pyomo model)
        2) Initialize the binary variables in the model with initial binary solutions
        3) Initialize other variables in the model, computed by the binary variable values

        Arguments
        ---------
        :param budget_opt: budget
        :param initial_solution: a dictionary, key: budget, value: initial solution pickle file name
            this option stores the available initial solutions with their budgets
        :param mixed_integer: boolean
            not relaxing integer decisions
        :param obj: Enum
            "A" or "D" optimality, use trace or determinant of FIM
        :param mix_obj: boolean
            if True, the objective function is a weighted sum of A- and D-optimality (trace and determinant)
        :param alpha: float
            range [0,1], weight of mix_obj. if 1, it is A-optimality. if 0, it is D-optimality
        :param fixed_nlp: boolean
            if True, the problem is formulated as a fixed NLP
        :param fix: boolean
            if solving as a square problem or with DOFs
        :param upper_diagonal_only: boolean
            if using upper_diagonal_only set to define decisions and FIM, or not
        :param num_dynamic_t_name: list
            a list of the exact time points for the dynamic-cost measurements time points
        :param budget: integer
            total budget
        :param init_cov_y: list of lists
            initialize decision variables
        :param initial_fim: list of lists
            initialize FIM
        :param dynamic_install_initial: list
            initialize if_dynamic_install
        :param total_measure_initial: integer
            initialize the total number of measurements chosen
        :param static_dynamic_pair: list of lists
            a list of the name of measurements, that are selected as either dynamic or static measurements.
        :param time_interval_all_dynamic: boolean
            if True, the minimal time interval applies for all dynamical measurements
        :param total_manual_num_init: integer
            initialize the total number of dynamical timepoints selected
        :param cost initial: float
            initialize the cost
        :param fim_diagonal_small_element: float
            a small number, default to be 0, to be added to FIM diagonal elements for better convergence
        :param print_level: integer
            0 (default): no extra printing
            1: print to show it is working
            2: print for debugging
            3: print everything that could help with debugging
        """

        # ==== Create the model ====
        # create model and save model to the object
        self._continuous_optimization(
            mixed_integer=mixed_integer,  # if relaxed binary variables
            fixed_nlp=fixed_nlp,  # if this is a fixed NLP problem
            fix=fix,  # if this is a square problem
            upper_diagonal_only=upper_diagonal_only,  # if we only define upper triangular for symmetric matrices
            num_dynamic_t_name=num_dynamic_t_name,  # exact time points for DCMs time points
            budget=budget_opt,  # budget
            init_cov_y=init_cov_y,  # initial cov_y values
            initial_fim=initial_fim,  # initial FIM values
            dynamic_install_initial=dynamic_install_initial,  # initial number of DCMs installed
            total_measure_initial=total_measure_initial,  # initial number of SCMs and time points selected
            static_dynamic_pair=static_dynamic_pair,  # if one measurement can be seen as both DCM and SCM
            time_interval_all_dynamic=time_interval_all_dynamic,  # time intervals between two time points of DCMs
            multi_component_pairs=multi_component_pairs,
            max_num_z=max_num_z,
            max_num_o=max_num_o,
            max_num_z_lists=max_num_z_lists,
            max_num_o_lists=max_num_o_lists,
            max_lists=max_lists,
            total_manual_num_init=total_manual_num_init,  # total time points of DCMs selected
            cost_initial=cost_initial,  # initial cost of the current solution
            fim_diagonal_small_element=fim_diagonal_small_element,  # a small element added to FIM diagonals
            print_level=print_level,
        )  # print level for optimization part

        # add objective function
        self._add_objective(
            obj=obj,  # objective function options, "A" or "D"
            mix_obj=mix_obj,  # if using a combination of A- and D-optimality
            alpha=alpha,  # if mix_obj = True, the weight of A-optimality
        )

        # store the initialization dictionary
        self.curr_res_list = (
            initial_solution  # dictionary that stores budget: solution file name
        )
        # ==== Initialize the model ====
        # locate the binary solution file according to the new budget
        initial_file_name = self._locate_initial_file(budget_opt)
        # initialize the model with the binary decision variables
        self._initialize_binary(initial_file_name)

        # store this for later use; or we need to input this again for update_budget
        self.obj = obj
        # warmstart function initializes the model with all the binary decisions values stored in the model
        if obj == ObjectiveLib.A:
            # use warmstart but without initializing grey-box block
            grey_box_opt = False
        else:
            # use warmstart with initializing grey-box block
            grey_box_opt = True

        # use warmstart with grey-box block
        self.customized_warmstart(grey_box=grey_box_opt)

    def update_budget(self, budget_opt):
        """
        Update the budget of the created model, then initialize the model corresponding to the budget

        Arguments
        ---------
        budget_opt: new budget
        """
        # update the budget
        self.mod.budget = budget_opt

        # update the initialization according to the new budget
        # locate the binary solution file according to the new budget
        initial_file_name = self._locate_initial_file(budget_opt)  # new budget
        # initialize the model with the binary decision variables
        self._initialize_binary(initial_file_name)

        # warmstart function initializes the model with all the binary decisions values stored in the model
        if self.obj == ObjectiveLib.A:
            # use warmstart but without initializing grey-box block
            grey_box_opt = False
        else:
            # use warmstart with initializing grey-box block
            grey_box_opt = True

        # use warmstart with grey-box block
        self.customized_warmstart(grey_box=grey_box_opt)

    def extract_store_sol(self, budget_opt, store_name):
        """
        Extract the solution from pyomo model, store in two pickles.
        First data file (store_name+"_fim_"+budget_opt): pickle, a Nm*Nm numpy array that is the solution of measurements
        Second data file (store_name+"_fim_"+budget_opt): pickle, a Np*Np numpy array that is the FIM

        Arguments
        ---------
        :param store_name: if not None, store the solution and FIM in pickle file with the given name
        """
        fim_result = np.zeros(
            (self.sens_info.n_parameters, self.sens_info.n_parameters)
        )
        for i in range(self.sens_info.n_parameters):
            for j in range(i, self.sens_info.n_parameters):
                fim_result[i, j] = fim_result[j, i] = pyo.value(
                    self.mod.total_fim[i, j]
                )

        print(fim_result)
        print("trace:", np.trace(fim_result))
        print("det:", np.linalg.det(fim_result))
        print(np.linalg.eigvals(fim_result))

        ans_y, _ = self.extract_solutions()
        print("pyomo calculated cost:", pyo.value(self.mod.cost))
        # print("if install dynamic measurements:")
        # print(pyo.value(self.mod.if_install_dynamic[3]))

        if store_name:

            file = open(store_name + str(budget_opt), "wb")
            pickle.dump(ans_y, file)
            file.close()

            file2 = open(store_name + "fim_" + str(budget_opt), "wb")
            pickle.dump(fim_result, file2)
            file2.close()

    def _locate_initial_file(self, budget):
        """
        Given the budget, select which initial solution it should use by locating the closest budget that has stored solutions
        It selects according to:
        1) if this budget has a previous solution, find it, return the file name
        2) if this budget does not have a previous solution, find the closest budget that has a solution, return the file name

        Arguments
        --------
        budget: budget

        Return
        ------
        y_init_file: the file name that contains a previous solution
        """
        ## find if there has been a original solution for the current budget
        if budget in self.curr_res_list:  # use an existed initial solutioon
            y_init_file = self.curr_res_list[budget]
            curr_budget = budget

        else:
            # if not, find the closest budget, and use this as the initial point
            curr_min_diff = np.inf  # current minimal budget difference
            curr_budget = max(list(self.curr_res_list.keys()))  # starting point

            # find the existing budget that minimize curr_min_diff
            for i in list(self.curr_res_list.keys()):
                # if we found an existing budget that is closer to the given budget
                if abs(i - budget) < curr_min_diff:
                    curr_min_diff = abs(i - budget)
                    curr_budget = i

        if self.precompute_print_level >= 1:
            print("using solution at", curr_budget, " too initialize")

        # assign solution file names, and FIM file names
        y_init_file = self.curr_res_list[curr_budget]

        return y_init_file

    def _initialize_binary(self, y_init_file):
        """This function initializes all binary variables directly in the created model

        Arguments
        ---------
        y_init_file: the file name that contains a previous solution

        Return
        ------
        None. self.mod is updated with initial values for all binary decisions
        """
        # read y
        with open(y_init_file, "rb") as f:
            init_cov_y = pickle.load(f)

        # Round possible float solution to be integer
        for i in range(self.num_measure_dynamic_flatten):
            for j in range(self.num_measure_dynamic_flatten):
                if init_cov_y[i][j] > 0.99:
                    init_cov_y[i][j] = int(1)
                else:
                    init_cov_y[i][j] = int(0)

        # initialize m.cov_y with the intial solution
        # loop over number of measurements
        for a in range(self.num_measure_dynamic_flatten):
            # cov_y only have the upper triangle part since solution matrix is symmetric
            for b in range(a, self.num_measure_dynamic_flatten):
                self.mod.cov_y[a, b] = init_cov_y[a][b]

        # initialize total manual number. This is not needed since it is initialized by warmstart function
        # kept for now
        # total_manual_init = 0

        # initialize the DCM installation flags
        # this needs initialized since it is binary decisions which warmstart function won't initialize
        dynamic_install_init = [0, 0, 0]

        # round solutions
        # if floating solutions, if value > 0.01, we count it as an integer decision that is 1 or positive
        for i in range(self.n_static_measurements, self.num_measure_dynamic_flatten):
            if init_cov_y[i][i] > 0.01:
                # total_manual_init += 1  # kept this line for now

                # identify which DCM this timepoint belongs to, turn the installation flag to be positive
                i_pos = int((i - self.n_static_measurements) / self.sens_info.Nt)
                dynamic_install_init[i_pos] = 1

        # initialize m.if_install_dynamic with the value calculated
        # loop over DCM index
        for i in self.mod.dim_dynamic:
            self.mod.if_install_dynamic[i] = dynamic_install_init[
                i - self.n_static_measurements
            ]

    def customized_warmstart(self, grey_box=True):
        """
        This is the warmstart function provided to MindtPy
        It is called every mindtpy iteration, after solving MILP master problem, before solving the fixed NLP problem
        This function initializes all continuous variables in the model given the integer decision values

        Return
        -----
        None. Initialize all continuous variables of this model in-place.
        """
        # new_fim is the FIM computed with the given integer solution
        # initialize new_fim
        new_fim = np.zeros((self.sens_info.n_parameters, self.sens_info.n_parameters))

        if self.optimize_print_level >= 3:
            # print to see how many non-zero solutions are in the solution
            for a in self.mod.n_responses:
                for b in self.mod.n_responses:
                    if b >= a:
                        # all solutions that are bigger than 0.001 are treated as non-zero
                        if self.mod.cov_y[a, b].value > 0.001:
                            print(a, b, self.mod.cov_y[a, b].value)

        # compute FIM
        def eval_fim_warmstart(a, b):
            """
            Evaluate FIM
            FIM = sum(cov_y[i,j]*unit FIM[i,j]) for all i, j in n_responses
            Only compute the upper triangle part since FIM is symmetric

            a, b: indices for FIM, iterate in parameter set
            """
            # Only compute the upper triangle part since FIM is symmetric
            if a <= b:
                # initialize the element
                summi = 0
                # loop over the number of measurements
                for i in self.mod.n_responses:
                    # loop over the number of measurements
                    for j in self.mod.n_responses:
                        # cov_y is also a symmetric matrix, we use only the upper triangle of it to compute FIM
                        if j >= i:
                            summi += (
                                self.mod.cov_y[i, j].value
                                * self.unit_fims[
                                    i * self.num_measure_dynamic_flatten + j
                                ][a][b]
                            )
                        else:
                            summi += (
                                self.mod.cov_y[j, i].value
                                * self.unit_fims[
                                    i * self.num_measure_dynamic_flatten + j
                                ][a][b]
                            )
                return summi
            # check error. we only compute upper diagonal
            else:
                raise ValueError(
                    "Only the upper diagonal matrix can be used for computation. "
                )

        # compute each element in FIM
        # loop over number of parameters
        for a in self.mod.dim_fim:
            # loop over number of parameters
            for b in self.mod.dim_fim:
                # Only compute the upper triangle part since FIM is symmetric
                if a <= b:
                    # compute FIM[a,b]
                    dynamic_initial_element = eval_fim_warmstart(a, b)
                    # FIM is symmetric, FIM[a,b] == FIM[b,a]
                    new_fim[a, b] = dynamic_initial_element
                    new_fim[b, a] = dynamic_initial_element

        if self.precompute_print_level >= 2:
            print("warmstart FIM computed by integer decisions:", new_fim)

        # initialize determinant
        # use slogdet to avoid ill-conditioning issue
        _, det = np.linalg.slogdet(new_fim)

        # if the FIM computed is a rank-deficient matrix, initialize with an identity matrix
        if det <= -10:
            if self.optimize_print_level >= 1:
                print(
                    "warmstart determinant is too small:",
                    det,
                    ". Use an identity matrix as FIM",
                )
            # update FIM used
            new_fim = np.zeros(
                (self.sens_info.n_parameters, self.sens_info.n_parameters)
            )
            # diagonal elements are 0.01
            for i in range(self.sens_info.n_parameters):
                new_fim[i, i] = 0.01

        # add the FIM small diagonal element to FIM to be consistent
        for i in range(self.sens_info.n_parameters):
            new_fim[i][i] += self.fim_diagonal_small_element

        # initialize grey-box module
        # loop over parameters
        for a in self.mod.dim_fim:
            # loop over parameters
            for b in self.mod.dim_fim:
                # grey-box only needs the upper triangle elements for it only defines and flattens the upper half
                if a <= b:
                    self.mod.total_fim[a, b].value = new_fim[a][b]

                    # if D-optimality:
                    if grey_box:
                        # grey-box uses a tuple as the input name
                        grey_box_name = (a, b)
                        # initialize grey-box value
                        self.mod.my_block.egb.inputs[grey_box_name] = new_fim[a][b]

        # initialize determinant
        # use slogdet to avoid ill-conditioning issue
        _, logdet = np.linalg.slogdet(new_fim)

        # if D-optimality
        if grey_box:
            # initialize grey-box module output
            self.mod.my_block.egb.outputs["log_det"] = logdet

        if self.optimize_print_level >= 1:
            print("Warmstart initialize FIM with: ", new_fim)
            print("Warmstart logdet:", logdet)
            print("Warmstart eigen value:", np.linalg.eigvals(new_fim))

        # manual and dynamic install initial
        def total_dynamic_exp(m):
            """compute the number of time points selected in total, to initialize total_number_dynamic_measurements
            This value is for initializing the total manual number constraint
            """
            return sum(
                m.cov_y[i, i].value
                for i in range(
                    self.n_static_measurements, self.num_measure_dynamic_flatten
                )
            )

        def total_exp(m):
            """compute the number of all measurements selected in total, to initialize total_number_measurements
            This value is for initializing integer cut
            """
            return sum(
                m.cov_y[i, i].value for i in range(self.num_measure_dynamic_flatten)
            )

        ### cost constraints
        def cost_compute(m):
            """Compute cost
            cost = static-cost measurement cost + dynamic-cost measurement installation cost + dynamic-cost meausrement timepoint cost
            """
            return sum(
                m.cov_y[i, i].value * self.cost_list[i]
                for i in range(self.num_measure_dynamic_flatten)
            ) + sum(
                m.if_install_dynamic[j].value
                * self.dynamic_install_cost[j - self.n_static_measurements]
                for j in m.dim_dynamic
            )

        # compute cost
        cost_init = cost_compute(self.mod)
        # copmute total number of dynamic time points selected
        total_dynamic_initial = total_dynamic_exp(self.mod)
        # compute total number of measurements selected
        total_initial = total_exp(self.mod)

        # initialize model variables
        self.mod.total_number_dynamic_measurements.value = total_dynamic_initial
        self.mod.total_number_measurements.value = total_initial
        self.mod.cost.value = cost_init

        if self.optimize_print_level >= 1:
            print("warmstart initialize total measure:", total_initial)
            print("warmstart initialize total dynamic: ", total_dynamic_initial)
            print("warmstart initialize cost:", cost_init)

    def continuous_optimization_cvxpy(self, objective="D", budget=100, solver=None):
        """
        This optimization problem can also be formulated and solved in the CVXPY framework.
        This is a generalization code for CVXPY problems for reference, not currently used for the paper.

        Arguments
        ---------
        :param objective: can choose from 'D', 'A', 'E' for now. if defined others or None, use A-optimality.
        :param cost_budget: give a total limit for costs.
        :param solver: default to be MOSEK. Look for CVXPY document for more solver information.

        Returns
        -------
        None
        """

        # compute Atomic FIM
        self.assemble_unit_fims()

        # evaluate fim in the CVXPY framework
        def eval_fim(y):
            """Evaluate FIM from y solution
            FIM = sum(cov_y[i,j]*unit FIM[i,j]) for all i, j in n_responses
            """
            fim = sum(
                y[i, j] * self.unit_fims[i * self.total_no_measure + j]
                for i in range(self.total_no_measure)
                for j in range(self.total_no_measure)
            )
            return fim

        def a_opt(y):
            """A-optimality as OBJ."""
            fim = eval_fim(y)
            return cp.trace(fim)

        def d_opt(y):
            """D-optimality as OBJ"""
            fim = eval_fim(y)
            return cp.log_det(fim)

        def e_opt(y):
            """E-optimality as OBJ"""
            fim = eval_fim(y)
            return -cp.lambda_min(fim)

        # construct variables
        y_matrice = cp.Variable(
            (self.total_no_measure, self.total_no_measure), nonneg=True
        )

        # cost limit
        p_cons = [
            sum(y_matrice[i, i] * self.cost[i] for i in range(self.total_no_measure))
            <= budget
        ]

        # loop over all measurement index
        for k in range(self.total_no_measure):
            # loop over all measurement index
            for l in range(self.total_no_measure):
                # y[k,l] = y[k]*y[l] relaxation
                p_cons += [y_matrice[k, l] <= y_matrice[k, k]]
                p_cons += [y_matrice[k, l] <= y_matrice[l, l]]
                p_cons += [y_matrice[k, k] + y_matrice[l, l] - 1 <= y_matrice[k, l]]
                p_cons += [y_matrice.T == y_matrice]

        # D-optimality
        if objective == "D":
            obj = cp.Maximize(d_opt(y_matrice))
        # E-optimality
        elif objective == "E":
            obj = cp.Maximize(e_opt(y_matrice))
        # A-optimality
        else:
            if self.verbose:
                print("Use A-optimality (Trace).")
            obj = cp.Maximize(a_opt(y_matrice))

        problem = cp.Problem(obj, p_cons)

        if not solver:
            problem.solve(verbose=True)
        else:
            problem.solve(solver=solver, verbose=True)

        self.__solution_analysis(y_matrice, obj.value)

    def extract_solutions(self):
        """
        Extract and show solutions from a solved Pyomo model
        mod is an argument because we

        Arguments
        --------
        mod: a solved Pyomo model

        Return
        ------
        ans_y: a numpy array containing the choice for all measurements
        sol_y: a Nd*Nt numpy array, each row contains the choice for the corresponding DCM at every timepoint
        """
        # ans_y is a numpy array of the shape Nm*Nm
        ans_y = np.zeros(
            (self.num_measure_dynamic_flatten, self.num_measure_dynamic_flatten)
        )

        # loop over the measurement choice index
        for i in range(self.num_measure_dynamic_flatten):
            # loop over the measurement choice index
            for j in range(i, self.num_measure_dynamic_flatten):
                cov = pyo.value(self.mod.cov_y[i, j])
                # give value to its symmetric part
                ans_y[i, j] = ans_y[j, i] = cov

        # round small errors to integers
        # loop over all measurement choice
        for i in range(len(ans_y)):
            # loop over all measurement choice
            for j in range(len(ans_y[0])):
                # if the value is smaller than 0.01, we round down to 0
                if ans_y[i][j] < 0.01:
                    ans_y[i][j] = int(0)
                # if it is larger than 0.99, we round up to 1
                elif ans_y[i][j] > 0.99:
                    ans_y[i][j] = int(1)
                # else, we keep two digits after decimal
                else:
                    ans_y[i][j] = round(ans_y[i][j], 2)

        for c in range(self.n_static_measurements):
            if ans_y[c, c] >= 0.5:
                print(self.measure_name[c], ": ", ans_y[c, c])

        # The DCM solutions can be very sparse and contain a lot of 0
        # so we extract it, group it by measurement
        sol_y = np.asarray(
            [
                ans_y[i, i]
                for i in range(
                    self.n_static_measurements, self.num_measure_dynamic_flatten
                )
            ]
        )
        # group DCM time points
        sol_y = np.reshape(sol_y, (self.n_dynamic_measurements, self.dynamic_Nt))
        np.around(sol_y)
        # loop over each DCM
        for r in range(len(sol_y)):
            # print(dynamic_name[r], ": ", sol_y[r])
            print(self.measure_name[r + self.n_static_measurements])
            # print the timepoints for the current DCM
            print(sol_y[r])
            # for i, t in enumerate(sol_y[r]):
            #    if t>0.5:
            #        print(self.dynamic_time[i])

        return ans_y, sol_y


def print_fim(fim):
    """
    Analyze one given FIM, this is a helper function after the optimization.

    Arguments
    ---------
    :param FIM: FIM matrix

    Returns
    -------
    None
    """

    det = np.linalg.det(fim)  # D-optimality
    trace = np.trace(fim)  # A-optimality
    eig = np.linalg.eigvals(fim)
    print("======FIM result======")
    print("FIM:", fim)
    print(
        "Determinant:",
        det,
        "; log_e(det):",
        np.log(det),
        "; log_10(det):",
        np.log10(det),
    )
    print(
        "Trace:",
        trace,
        "; log_e(trace):",
        np.log(trace),
        "; log_10(trace):",
        np.log10(trace),
    )
    print(
        "Min eig:",
        min(eig),
        "; log_e(min_eig):",
        np.log(min(eig)),
        "; log_10(min_eig):",
        np.log10(min(eig)),
    )
    print("Cond:", max(eig) / min(eig))
