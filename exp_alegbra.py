
def expectation_eval(Hst = [],count_z = None,count_x= None, shots = 100):
    
    tot_exp = 0
    for i in Hst:
        prefactor,op = i.split('*')[0],i.split('*')[1]
            #print('ham exp=',op)
        if op.find('Z')>=0:
                #print("here in Z")
            counts = count_z
        elif op.find('X')>=0:
                #print("here in X")
            counts = count_x
        exp = expectation_value(count = counts, operator= op, shots = shots)
            #print('Back=',i,exp*float(prefactor))
        tot_exp += float(prefactor)*exp
            
    return tot_exp


def active_bits(operator=''):
    hot_bits = []
    for i,op in enumerate(operator):
            #print(i,op)
        if op!='I':
            hot_bits.append(len(operator)-1 - i)
    return hot_bits

def expectation_value(count= None, operator ='', shots = None):
    active_qubits = active_bits(operator)
        #print(operator,active_qubits)
    count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
        #print('count =',count)
    tot = 0
    
    for key in count.keys():
            #print(key)
        num = 1
        for qbits in active_qubits:
            num  = num*(-1)**key[len(operator)-1-qbits]
            #print(num, num*count[key]) 
        tot += num*count[key]
        
    expectation_val = tot/shots
    return expectation_val