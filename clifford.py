
import numpy as np
from qiskit.circuit.library import (IGate, XGate, YGate, ZGate,
                                    HGate, SGate, SdgGate,
                                    SXGate, SXdgGate)

class CliffordSet():
    def __init__(self,
                 nlayers=0,
                 seed=0):
        self.nlayers = nlayers
        self.seed = seed

    def generate_clifford_set(self):
        """
        Creates the set of randomly drawn Clifford gates
        according to the number of ansatz layers
        """
        print("generate Clifford set\n")
        # Define the set of gates to randomly draw from
        single_qubit_clifford_group = (IGate, XGate, YGate, ZGate,
                                        HGate, SGate, SdgGate,
                                        SXGate, SXdgGate)

        # Use input seed
        np.random.seed(self.seed)

        # Draw Clifford gates
        return [np.random.choice(single_qubit_clifford_group)()
                for _ in range(self.nlayers * 20)]

