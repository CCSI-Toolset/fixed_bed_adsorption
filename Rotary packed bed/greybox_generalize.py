import numpy as np
import pyomo.environ as pyo
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

class LogDetModel(ExternalGreyBoxModel):
    """Greybox model to compute the log determinant of a matrix.
    """
    def __init__(self, n_parameters=2, initial_fim=None, use_exact_derivatives=True,verbose=True):
        """
        Parameters

        n_parameters: int 
            Number of parameters in the matrix.
        initial_fim: dict
            Initial value of the matrix. If None, the identity matrix is used.
        use_exact_derivatives: bool 
            If True, the exact derivatives are used. If False, the finite difference
            approximation is used.
        verbose: bool
            If True, print information about the model.
        """
        self._use_exact_derivatives = use_exact_derivatives
        self.verbose = verbose
        self.n_parameters = n_parameters
        self.num_input = int(n_parameters + (n_parameters*n_parameters-n_parameters)//2)
        self.initial_fim = initial_fim
        #print(initial_fim)
        #print("initialize with:", self.initial_fim)
        
        # For use with exact Hessian
        self._output_con_mult_values = np.zeros(1)

        if not use_exact_derivatives:
            raise NotImplementedError("use_exact_derivatives == False not supported")
        
    def input_names(self):
        """Return the names of the inputs."""
        input_name_list = []
        for i in range(self.n_parameters):
            for j in range(i,self.n_parameters):
                input_name_list.append("ele_"+str(i)+"_"+str(j))
                
        return input_name_list

    def equality_constraint_names(self):
        """ Return the names of the equality constraints."""
        # no equality constraints
        return [ ]
    
    def output_names(self):
        """ Return the names of the outputs."""
        return ['log_det']

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        """Set the values of the output constraint multipliers."""
        # because we only have one output constraint
        assert len(output_con_multiplier_values) == 1
        np.copyto(self._output_con_mult_values, output_con_multiplier_values)

    def finalize_block_construction(self, pyomo_block):
        """Finalize the construction of the ExternalGreyBoxBlock."""
        ele_to_order = {}
        count  = 0
        # initialize, set up LB and UB
        # only generating upper triangular part
        # loop over parameters
        for i in range(self.n_parameters):
            # loop over parameters from current parameter to end
            for j in range(i, self.n_parameters):
                # flatten (i,j)
                ele_to_order[(i,j)], ele_to_order[(j,i)] = count, count 
                str_name = 'ele_'+str(i)+"_"+str(j)
                
                if self.initial_fim is not None:
                    print("initialized")
                    pyomo_block.inputs[str_name].value = self.initial_fim[str_name]
                else:
                    print("uninitialized")
                    # identity matrix 
                    if i==j:
                        pyomo_block.inputs[str_name].value = 1
                    else:
                        pyomo_block.inputs[str_name].value = 0
                    
                count += 1 
                
        self.ele_to_order = ele_to_order

    def set_input_values(self, input_values):
        """Set the values of the inputs."""
        self._input_values = list(input_values)

    def evaluate_equality_constraints(self):
        """Evaluate the equality constraints."""
        # Not sure what this function should return with no equality constraints
        return None
    
    def evaluate_outputs(self):
        """Evaluate the output of the model."""
        # form matrix as a list of lists
        M = self._extract_and_assemble_fim()

        # compute log determinant
        (sign, logdet) = np.linalg.slogdet(M)

        if self.verbose:
            #print("\n Consider M =\n",M)
            #print(self._input_values)
            #print("   logdet = ",logdet,"\n")
            #print("Eigvals:", np.linalg.eigvals(M))
            print("iteration")

        return np.asarray([logdet], dtype=np.float64)

    def evaluate_jacobian_equality_constraints(self):
        """Evaluate the Jacobian of the equality constraints."""
        return None
    
    def _extract_and_assemble_fim(self):
        M = np.zeros((self.n_parameters, self.n_parameters))
        for i in range(self.n_parameters):
            for k in range(self.n_parameters):                
                M[i,k] = self._input_values[self.ele_to_order[(i,k)]]

        return M

    def evaluate_jacobian_outputs(self):
        """Evaluate the Jacobian of the outputs."""
        if self._use_exact_derivatives:
            M = self._extract_and_assemble_fim()

            # compute pseudo inverse to be more numerically stable
            Minv = np.linalg.pinv(M)

            # compute gradient of log determinant
            row = np.zeros(self.num_input) # to store row index
            col = np.zeros(self.num_input) # to store column index
            data = np.zeros(self.num_input) # to store data
            
            # construct gradients as a sparse matrix 
            # loop over parameters
            for i in range(self.n_parameters):
                # loop over parameters from current parameter to end
                for j in range(i, self.n_parameters):
                    order = self.ele_to_order[i,j]
                    # diagonal elements
                    if i==j: 
                        row[order], col[order], data[order] = (0,order, Minv[i,j])
                    # off-diagonal elements
                    else: # factor = 2 since it is a symmetric matrix
                        row[order], col[order], data[order] = (0,order, 2*Minv[i,j])
            # sparse matrix
            return coo_matrix((data, (row, col)), shape=(1, self.num_input))
  