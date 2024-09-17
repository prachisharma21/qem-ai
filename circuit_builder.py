
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.providers.fake_provider import FakeNairobiV2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit import Aer
from qiskit.quantum_info import SparsePauliOp 

from exp_alegbra import expectation_eval

class CircuitBuilder():
    """
    explain
    """

    def __init__(self,backend = FakeNairobiV2(), nlayers=1,cliffordset = [], circuit_type ='ideal',shots =2**18, Hstr = [],EM_gates =[]):
        """
        Parameters:
            nlayers: 

        """
        self.backend = backend
        self.num_qubits = self.backend.num_qubits
        self.nlayers = nlayers
        self.cliffordset =cliffordset
        self.shots = shots
        self.circuit_type = circuit_type
        self.EM_gates = EM_gates
        self.Hstr = Hstr
        # Make a copy of the Clifford set for popping
        self.tmp_cliffordset = self.cliffordset.copy()
        
        self.disjoint_entangling_pairs =[]
        if isinstance(self.backend, FakeNairobiV2):
            #self.disjoint_entangling_pairs = [[(0,1),(3,5)],[(1,2),(4,5)],[(1,3),(5,6)]]
            self.disjoint_entangling_pairs = [[(0,1),(4,5)],[(1,2),(3,5)],[(1,3),(5,6)]]
        
        
        self.ideal_circ = self.Qcircuit()
        
        self.noisy_circ  = self.Noisy_training_circuit(circuit=self.ideal_circ)
        self.noisy_circ_test = self._Noisy_training_circuit_test(circuit=self.ideal_circ)
        #print("Expectation value of the training circuit with only pauli twirling gates",self.noisy_expectation_test())
        

    def Qcircuit(self):
        """
        Creates the ideal quantum circuit in + initial state with HVA ansatz 
        """
        self.circ = QuantumCircuit(self.num_qubits)

        # Prepare the initial state in |+> state 
        for i in range(self.num_qubits):
            self.circ.h(i)

        # Add HVA ansatz layers 
        for _ in range(self.nlayers):
            #self._add_HVA_layer_test()
            self.add_HVA_layer()

        return self.circ


    def add_HVA_layer(self):
        """
        Apply a HVA layer to the circuit 
        """
        self.add_singleq_clifford_layer()
        self.add_singleq_clifford_layer()
        for bonds in self.disjoint_entangling_pairs:
            for bond in bonds:
                gate = self.tmp_cliffordset.pop()
                self.circ.cx(bond[0],bond[1])
                self.circ.append(gate,[bond[1]])
                self.circ.cx(bond[0],bond[1])
        #self.circ.barrier()

        
    def _add_HVA_layer_test(self):
        """
        Helper test function to create a test circuit compare the expectation values and do checks!!
        """
        for bonds in self.disjoint_entangling_pairs:
            for bond in bonds:
                self.circ.cx(bond[0],bond[1])
                self.circ.h(bond[0])
                self.circ.rz(np.pi,bond[1])
                self.circ.cx(bond[0],bond[1])   

    def add_singleq_clifford_layer(self):
        """
        Apply a single qubit Clifford gate layer
        """
        for i in range(self.num_qubits):
            gate = self.tmp_cliffordset.pop()
            self.circ.append(gate,[i])


    def Qcircuit_to_benchmark_layers(self,layer = None):
        """
        Breaks a circuit into layers following the pattern - 
            -> Single qubit gates + disjoint layer of self-adjoint Clifford gates
        input: self.circ
        output: layers (CircuitLayer)

        """

        layers = []
        qc = layer
    
        inst_list = [inst for inst in qc if not inst.operation.name=='measure'] 

        #pop off instructions until inst_list is empty
        while inst_list:

            circ = qc.copy_empty_like() #blank circuit to add instructions
            #print(circ)
            layer_qubits = set() #qubits in the support of two-qubit clifford gates
            twirl_layer = qc.copy_empty_like()        

            for inst in inst_list.copy(): #iterate through remaining instructions

                #check if current instruction overlaps with support of two-qubit gates
                #already on layer
                
                if not layer_qubits.intersection(inst.qubits):
                    circ.append(inst)
                    inst_list.remove(inst)

                if len(inst[1]) == 2:                   
                    layer_qubits = layer_qubits.union(inst.qubits) #add support to layer

                
            newlayer = circ
            # layer_qubits are not doing much at the moment..circ is enough
            if self.separate_gates(layer=newlayer,weight=2): #append only if not empty
                layers.append(circ)
        return layers
    
    def separate_gates(self,layer =None, weight =1):
        """This method parses the list of gates in the input layer and returns a Circuit
        consisting only of gates with the desired weight"""

        qc =layer.copy_empty_like() 
        for inst in layer:
            if inst.operation.name in ['barrier','measure']:
                continue
            if len(inst.qubits) == weight:
                qc.append(inst)
        return qc
    
    def save_pauli_twirl_str(self,fname = '',lines = None):
        with open(fname,'w+') as f:
            for line in lines:
                f.write(f"{line}\n")

        
    
    def apply_pauli_twirl_and_EM_gates(self,layers = None, add_EM_gates = False):
        """
        Helper function to apply Pauli twirl and optional error mitigating P gates to each layer.
    
        Parameters:
            layers (list): The layers of the circuit.
            add_EM_gates (bool): Whether to apply the EM gates or not.
    
        Returns:
            list: The modified layers with Pauli twirl and EM gates (if specified).
        """

        p_list = ['I','X','Y','Z']
        np.random.seed(0)
        
        pauli_twirl_sequences = [np.random.choice(p_list,len(lay.qubits)) for lay in layers]
        twirl_layers = []
        list_pauli_strings = []       
        
        for id,lay in enumerate(layers):           
            pauli_twirl = pauli_twirl_sequences[id]
            
            single_q_gates = self.separate_gates(layer=lay,weight=1)
            two_q_gates  = self.separate_gates(layer=lay,weight=2)
            
            qc = lay.copy_empty_like()
            qc = qc.compose(single_q_gates)
            #qc.barrier()
            
            # Add left EM gates if requested
            if add_EM_gates:
                left_pstr,right_pstr  = self.EM_gates["EM_gates"][id]            
                qc = self.add_pauli_sequence(circuit=qc,seq=left_pstr)
        
            #qc.barrier()
            # Apply Pauli twirl sequence
            qc = self.add_pauli_sequence(circuit=qc,seq="".join([i for i in pauli_twirl]))
            
            # Apply the two-qubit gates
            qc = qc.compose(two_q_gates)
            
            # Conjugate the Pauli twirl sequence and apply
            conj_pauli_twirl = self.conjugate(pauli=Pauli("".join([i for i in pauli_twirl])),layer=two_q_gates)
            qc = self.add_pauli_sequence(circuit=qc,seq=str(conj_pauli_twirl))
            
            # To save the pauli twirl gates
            list_pauli_strings.append(("".join([i for i in pauli_twirl]),str(conj_pauli_twirl)))
            
            # Add right EM gates if requested
            if add_EM_gates:
                qc = self.add_pauli_sequence(circuit=qc,seq=right_pstr)
            #qc.barrier()
            
            twirl_layers.append(qc)
        
        self.save_pauli_twirl_str('pauli_twirl_str.txt',list_pauli_strings)

        return twirl_layers  

    def add_pauli_twirl_only(self,layers = None):
        """
        To build test circuit with only Pauli twirling to the ideal clifford circuit 
        
        """
        return self.apply_pauli_twirl_and_EM_gates(layers=layers, add_EM_gates= False)
    
    def add_pauli_twirl_and_EM_gates(self, layers=None):
        """
        Adds Pauli twirling and EM gates to the circuit layers.
        """
        return self.apply_pauli_twirl_and_EM_gates(layers=layers, add_EM_gates=True)

    
    def _Noisy_training_circuit_test(self,circuit = None):
        """
        Test training circuit with only Pauli twirling--no state-pre/readout/EM gates applied
        """
        ideal_circ = circuit
        noisy_circ = ideal_circ.copy_empty_like()
        
        layers = self.Qcircuit_to_benchmark_layers(layer=ideal_circ)
        twirled_layers =  self.add_pauli_twirl_only(layers=layers)
        for i in twirled_layers:
            noisy_circ = noisy_circ.compose(i)
        
        return noisy_circ
        
    def add_pauli_sequence(self, circuit = None,seq=None):
        for id,gate in enumerate(Pauli(seq)):
               circuit.append(gate,[id])
        return circuit
        
    
    def Noisy_training_circuit(self,circuit = None):
        """
        Creates a training circuit with state-prep, EM gates, pauli_twirling, and readout gates. 
        Parameters:
            circuit (Quantum circuit): the ideal circuit on which we want to create training circuit of.          
        Returns:
            circuit (Quantum circuit): the training circuit 
        """
        ideal_circ = circuit
        noisy_circ = ideal_circ.copy_empty_like()
        # apply the state-prep sequence of pauli-gates 
        state_prep, readout = self.EM_gates["state_prep_readout"]
        noisy_circ = self.add_pauli_sequence(circuit=noisy_circ,seq = state_prep)
        #noisy_circ.barrier()
        # create a circuit in terms of layers of disjoint two-qubits gates 
        layers = self.Qcircuit_to_benchmark_layers(layer=ideal_circ)
        # apply pauli twirling and Error-mitigating P gates to each layer 
        twirled_layers =  self.add_pauli_twirl_and_EM_gates(layers=layers)

        # create a final training circuit by looping over these layers with twirling and EM gates 
        for i in twirled_layers:
            noisy_circ = noisy_circ.compose(i)
        # apply the read-out pauli sequence
        noisy_circ = self.add_pauli_sequence(circuit=noisy_circ,seq = readout)
        
        return noisy_circ
    
    
    def nophase(self,pauli):
        """remove the phase from a Pauli"""
        return Pauli((pauli.z, pauli.x))

    def conjugate(self,pauli,layer=None):
        """It gives the Pdagger for noise twirling"""
        return self.nophase(pauli.evolve(layer))
    

    def Qcircuit_w_meas(self, circuit=None):
        """
        Prepares two measurement circuits: one for Z-basis and one for X-basis.
        """

        self.qc_z_meas = self._prepare_measurement_circuit(circuit, basis='Z')
        self.qc_x_meas = self._prepare_measurement_circuit(circuit, basis='X')

    def _prepare_measurement_circuit(self, circuit: QuantumCircuit, basis: str = 'Z') -> QuantumCircuit:
        """
        Helper function to prepare a measurement circuit in the specified basis.
    
        Parameters:
            circuit (QuantumCircuit): The base circuit to copy.
            basis (str): Either 'Z' for Z-basis measurement or 'X' for X-basis measurement.
        
        Returns:
            QuantumCircuit: The measurement circuit.
        """
        qc_meas = circuit.copy()
        
        if basis == 'X':
            # Apply Hadamard gates to measure in the X basis
            for i in range(self.num_qubits):
                qc_meas.h(i)

        # Apply measurements to all qubits
        qc_meas.measure_all()

        return qc_meas

            
    def get_meas_counts(self,circ= None, simulator = None):
        """
        Calculates the counts by transpiling the circuit and return the counts for the respective simulator
        """
        transpiled = transpile(circ,simulator)
        #print(transpiled)
        counts = simulator.run(transpiled,shots = self.shots).result().get_counts()
        return counts 
    
  
    
    def ideal_expectation(self):
        """
        Calculates the ideal expectation value using the clifford simulator Stabalizer simulator
        """
        self.Qcircuit_w_meas(circuit=self.ideal_circ)   
        count_x = self.get_meas_counts(circ = self.qc_x_meas, simulator=AerSimulator(method="stabilizer"))  
        count_z = self.get_meas_counts(circ = self.qc_z_meas, simulator=AerSimulator(method="stabilizer")) 
        return expectation_eval(Hst = self.Hstr,count_z=count_z,count_x=count_x, shots = self.shots)
    
    def noisy_expectation(self):
        """
        Calculates the expectation value using the training circuit on the noisy backend 
        """        
        self.Qcircuit_w_meas(circuit=self.noisy_circ)      
        count_x_noisy = self.get_meas_counts(circ = self.qc_x_meas, simulator=self.backend)  
        count_z_noisy = self.get_meas_counts(circ = self.qc_z_meas, simulator=self.backend) 
        return expectation_eval(Hst = self.Hstr,count_z=count_z_noisy,count_x=count_x_noisy, shots = self.shots)

    def _noisy_expectation_test(self): 
        """
        Calculates expectation value using the ideal circuit with only pauli twirling on the noisy backend
        """       
        self.Qcircuit_w_meas(circuit=self.noisy_circ_test)      
        count_x_noisy = self.get_meas_counts(circ = self.qc_x_meas, simulator=self.backend)  
        count_z_noisy = self.get_meas_counts(circ = self.qc_z_meas, simulator=self.backend) 
        return expectation_eval(Hst = self.Hstr,count_z=count_z_noisy,count_x=count_x_noisy, shots = self.shots)
    
    def _ideal_expectation_test(self):
        """
        Calculates the expectation value using the ideal circuit on the noisy backend
        """
        self.Qcircuit_w_meas(circuit=self.ideal_circ)   
        count_x = self.get_meas_counts(circ = self.qc_x_meas, simulator=self.backend)  
        count_z = self.get_meas_counts(circ = self.qc_z_meas, simulator=self.backend) 
        return expectation_eval(Hst = self.Hstr,count_z=count_z,count_x=count_x, shots = self.shots)

    
    def State_Vector_Simulator_check(self):
        """
        Calculates the expectation value using the ideal circuit on the state-vector simulator
        """
        count = Aer.get_backend('statevector_simulator').run(self.circ).result().get_statevector()
        count_noisy = Aer.get_backend('statevector_simulator').run(self.noisy_circ).result().get_statevector()
        
        sv = 0
        sv_noisy = 0
        for i in self.Hstr:
            opp = SparsePauliOp([i.split('*')[1] ],coeffs=[i.split('*')[0]]) #
            val =count.expectation_value(opp)
            val_noisy =count_noisy.expectation_value(opp)
            sv+=val
            sv_noisy+=val_noisy
        
        return np.real(sv), np.real(sv_noisy)
    
            


       






