#!/opt/python-2.6/bin/python2.6
# Mami Hackl and Nat Byington
# LING 572 HW8 Q2 
# Classify test data using a SVM model created by libSVM
# Args: test, model, sys_out 

import sys
import re
import math

## Classes
class Test_Vector:
    ''' An object representing a single instance from the test file. '''

    def __init__(self, true_class, features):
        self.true_class = true_class # 0 or 1; type Int
        self.features = features #set of feature indices associated with vector
    
class Model_Vector:
    ''' An object representing a single vector from the model file. '''
    
    def __init__(self, weight, features):
        self.weight = weight
        self.features = features #set of feature indices associated with vector
          
## Functions
def dot_product(model_vector, test_vector):
    ''' Return the dot product of two feature vector sets.
        Important: this function assumes that feature values are binary.'''
    intersection = test_vector.features & model_vector.features
    return float(len(intersection))
    
def get_kernel_function(k_name):
    ''' Return appropriate function based on kernel name. Default is
        linear.'''
    if k_name == 'polynomial':
        return poly_k
    if k_name == 'rbf':
        return rbf_k
    if k_name == 'sigmoid':
        return sig_k
    else:
        return lin_k
        
def lin_k(model_v, test_v):
    ''' Linear kernel, merely dot product.'''
    return dot_product(model_v, test_v)
    
def poly_k(model_v, test_v, gamma, coef, degree):
    ''' Polynomial kernel function.'''
    return (gamma * dot_product(model_v, test_v) + coef)**degree
    
def rbf_k(model_v, test_v, gamma):
    ''' RBF kernel function. '''
    diff = len(test_v.features - model_v.features) # difference of sets
    x = -1 * gamma * (diff**2)
    return math.exp(x)
    
def sig_k(model_v, test_v, gamma, coef):
    ''' Sigmoid kernel function. '''
    x = gamma * dot_product(model_v, test_v) + coef
    return math.tanh(x)
    

## Main
# Open arg files
test_file = open(sys.argv[1])
model_file = open(sys.argv[2])
sys_out = open(sys.argv[3], 'w')

# Variables
test_vectors = [] # list of test vectors
model_vectors = [] # list of model vectors
kernel_name = '' # will get set to kernel used in model file
k_func = False # will be the appropriate kernel function
func_args = [] # will be a list of additional args to pass to kernel function
rho = 0.0
correct = 0 # a counter used to calculate classification accuracy

# Process test file
for line in test_file.readlines():
    t_class = int(re.match('^(\d)', line).group(1))
    f_set = set(re.findall(' (\d+):1', line)) # set of feature indices as strings
    v = Test_Vector(t_class, f_set)
    test_vectors.append(v)
    
# Gather info from model using ugly if-statements; hard-coded for goodness
model = model_file.readlines()
kernel_name = re.match('kernel_type (\w+)', model[1]).group(1) # 2nd line of model
k_func = get_kernel_function(kernel_name)

if kernel_name == 'linear':
    sv_count = int(re.match('total_sv (\d+)', model[3]).group(1))
    rho = float(re.match('rho (\S+)', model[4]).group(1))
    sv_index = 8
if kernel_name == 'polynomial':
    sv_count = int(re.match('total_sv (\d+)', model[6]).group(1))
    rho = float(re.match('rho (\S+)', model[7]).group(1))
    sv_index = 11
    degree = float(re.match('degree (\S+)', model[2]).group(1))
    gamma = float(re.match('gamma (\S+)', model[3]).group(1))
    coef = float(re.match('coef0 (\S+)', model[4]).group(1))
    func_args = [gamma, coef, degree] # order is important!!
if kernel_name == 'rbf':
    sv_count = int(re.match('total_sv (\d+)', model[4]).group(1))
    rho = float(re.match('rho (\S+)', model[5]).group(1))
    sv_index = 9
    gamma = float(re.match('gamma (\S+)', model[2]).group(1))
    func_args = [gamma]
if kernel_name == 'sigmoid':
    sv_count = int(re.match('total_sv (\d+)', model[5]).group(1))
    rho = float(re.match('rho (\S+)', model[6]).group(1))
    sv_index = 10
    gamma = float(re.match('gamma (\S+)', model[2]).group(1))
    coef = float(re.match('coef0 (\S+)', model[3]).group(1))
    func_args = [gamma, coef] # order is important!!

# Gather model vectors
for i in range(sv_index, (sv_index + sv_count)):
    weight = float(re.match('^(\S+) ', model[i]).group(1))
    f_set = set(re.findall(' (\d+):1', model[i])) # set of feature indices as strings
    v = Model_Vector(weight, f_set)
    model_vectors.append(v)
    
# Classify test vectors; output results to sys_out
for test_v in test_vectors:
    sys_class = False
    f_x = 0.0 - rho
    for model_v in model_vectors:
        args_list = [model_v, test_v]
        args_list.extend(func_args) # add kernel-specific args
        args = tuple(args_list)
        f_x += model_v.weight * k_func(*args) # args gets expanded
    if f_x >= 0:
        sys_class = 0
    else:
        sys_class = 1
    if sys_class == test_v.true_class:
        correct += 1
    output = [str(test_v.true_class), str(sys_class), str(f_x), '\n']
    sys_out.write(' '.join(output))
    
# Output accuracy
acc = correct / float(len(test_vectors))
print acc

