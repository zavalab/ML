# -*- coding: utf-8 -*-
"""
calc_tools_no_mdtraj.py
This code contains all calc tools without mdtraj installed


Created on Thu Apr 18 08:51:54 2019

@author: akchew
"""
### FUNCTION THAT SPLITS A LIST INTO MULTIPLE PARTS
def split_list(alist, wanted_parts=1):
    '''
    The purpose of this function is to split a larger list into multiple parts
    INPUTS:
        alist: [list] original list
        wanted_parts: [int] number of splits you want
    OUTPUTS:
        List containing chunks of your list
    Reference: https://stackoverflow.com/questions/752308/split-list-into-smaller-lists?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    '''
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]
