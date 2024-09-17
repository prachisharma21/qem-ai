from qiskit.providers.fake_provider import FakeNairobiV2
from circuit_builder import CircuitBuilder
import numpy as np 
from clifford import CliffordSet
from collections import defaultdict
import json

def random_pauli_str():
    p_list = ['I','X','Y','Z']
    p_str = np.random.choice(p_list,7)
    return "".join([i for i in p_str])
    
def error_mitigating_pauli_gates():
    return (random_pauli_str(),random_pauli_str())

def EM_gate_layer(entangling_layers = 6):
    EM_layer = defaultdict()
    EM_gates = []
    for i in range(entangling_layers):
        tup = error_mitigating_pauli_gates()
        EM_gates.append(tup)
    EM_layer["EM_gates"] = EM_gates
    EM_layer["state_prep_readout"] = (random_pauli_str(),random_pauli_str())
    with open('EM_pauli_layers.txt','w+') as f:
        f.write(json.dumps(EM_layer))
    return EM_layer

# Generate a set of Clifford gates
Hstr = ["-1.0*XIIIIII",
        "-1.0*IXIIIII",
        "-1.0*IIXIIII",
        "-1.0*IIIXIII",
        "-1.0*IIIIXII",
        "-1.0*IIIIIXI",
        "-1.0*IIIIIIX",
        "-1.0*IIIIIZZ",
        "-1.0*IZZIIII",
        "-1.0*IIIIZZI",
        "-1.0*ZZIIIII",
        "-1.0*IZIZIII",
        "-1.0*IIIZIZI",
        "0.5*ZIIIIII",
        "0.5*IZIIIII",
        "0.5*IIZIIII",
        "0.5*IIIZIII",
        "0.5*IIIIZII",
        "0.5*IIIIIZI",
        "0.5*IIIIIIZ"] 

clifford_set = CliffordSet(nlayers=1, seed = 10).generate_clifford_set()
#print(clifford_set)
circuit = CircuitBuilder(backend=FakeNairobiV2(),nlayers=1,cliffordset=clifford_set,shots=5*10**5,Hstr=Hstr, EM_gates = EM_gate_layer(6))

# uncomment below to print and visualize the circuit 
#print(circuit.ideal_circ)
#print(circuit.noisy_circ.draw())

# expectation evaluated below is for a ideal circuit using stabalizer simulator 
print(f'expectation of the ideal circuit  = {circuit.ideal_expectation()}')
# expectation evaluated below is for a training circuit with state-prep & readout & P (EM-gates) applied using noisy fake backend 
print(f'noisy expectation of the training circuit = {circuit.noisy_expectation()}')

# the rest are some test done 
print("Just for check")
print(f"Expectation value on Noisy backend \n For the ideal circuit: {circuit._ideal_expectation_test()}, \n For training circuit with only Pauli twirling gates: {circuit._noisy_expectation_test()}")
sv_results = circuit.State_Vector_Simulator_check()
print(f"State-vector calculations: \n expectation of ideal circuit com^EF(R,I) = {sv_results[0]}, \n expectation of training circuit for a set of P gates, i.e., com(R,P) = {sv_results[1]}")
print("com^EM(R,I) = sum_P q(P) com(R,P), and Error = |com^EM(R,I) - com^EF(R,I)| ")


        

        