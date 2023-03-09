# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:25:19 2021

@author: sqin34
"""

import numpy as np
import gudhi as gd

def getEC(data,numPoints,filtrations):
    cubeplex = gd.CubicalComplex(dimensions = [np.shape(data)[0],np.shape(data)[1]],top_dimensional_cells=np.ndarray.flatten(data));
    cubeplex.persistence();
    b = np.zeros( (numPoints,2) ) ;
    ec = np.zeros(numPoints);
    for (i,fval) in enumerate(filtrations):
        betti = cubeplex.persistent_betti_numbers(fval,fval);
        b[i] = [betti[0], betti[1]];
        ec[i] = betti[0] - betti[1];
    return ec