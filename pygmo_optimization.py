# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:25:06 2021

@author: Utkarsh
"""
import pygmo as pg

class simulation:
    def fitness(x):
        M = x[0]+x[1]
        L = x[1]
        r = simulate_Periodic(300,M,L)
        return np.mean(r[0])