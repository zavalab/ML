# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:44:17 2020

@author: sqin34
"""


import numpy as np

def AB_linear(length):
    synthdat = []
    for i,lenA in enumerate(range(length // 2 + 1, length)):
        tempA = 'A' * lenA
        tempB = 'B' * (length - lenA)
        synthdat.append(tempA + tempB)
    return synthdat
#test = AB_linear(10)

def AB_branch(length,branch_type):
    temp_AB = AB_linear(length)
    randomIndex = np.random.randint(2,length,len(temp_AB))
    for i,each_AB in enumerate(temp_AB):
        temp_AB[i] = each_AB[:randomIndex[i]] + '('+branch_type+')' + each_AB[randomIndex[i]:]
    return temp_AB
#test = AB_branch(10,'C')

def AB_multibranch(length,branch_type,num_branch):
    temp_AB = AB_linear(length)
    branch = '('+branch_type+')'
    randomIndex = np.random.randint(2,length,(len(temp_AB),num_branch))
    randomIndex.sort(axis=1)
    for k in range(1,num_branch):
        randomIndex[:,k] = randomIndex[:,k] + k*len(branch)
    for j in range(num_branch):
        for i,each_AB in enumerate(temp_AB):
            temp_AB[i] = each_AB[:randomIndex[i,j]] + branch + each_AB[randomIndex[i,j]:]
    return temp_AB
#test = AB_multibranch(10,'C',1)
#test = AB_multibranch(20,'CD',2)
    

def A_linear(min_len,max_len,char='A'):
    synthdat = []
    for i in range(min_len,max_len):
        temp = char * i
        synthdat.append(temp)
    return synthdat

def add_ring(alkyl_list,ring_length,ring_atom='C',random_pos=True):
    synthdat = []
    ring = ring_atom + '1' + ring_atom * (ring_length-1) + '1'
    if random_pos==True:
        for i,each_A in enumerate(alkyl_list):
            randomIndex = np.random.randint(len(each_A))
            synthdat.append(each_A[:randomIndex] + ring + each_A[randomIndex:])
    else:
        for i,each_A in enumerate(alkyl_list):
            synthdat.append(each_A + ring)
    return synthdat

def add_branch(alkyl_list,branch_type):
    synthdat = []
    for i,each_A in enumerate(alkyl_list):
        randomIndex = np.random.randint(2,len(each_A))
        while each_A[randomIndex] == 'O':
            randomIndex = np.random.randint(2,len(each_A))
        synthdat.append(each_A[:randomIndex+1] + branch_type + each_A[randomIndex+1:])
    return synthdat

#def A_glucoside(alkyl_list,junction_atom='O'):
#    synthdat = []
#    ring = junction_atom + 'C1OC(CO)C(O)C(O)C1O'
        
