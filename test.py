# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:33:29 2021

@author: Utkarsh


"""


from pyDOE2 import ff2n, bbdesign, ccdesign, fullfact
import numpy  as np
from funcs import *

np.random.seed(0)

'''
print(np.unique(ccdesign(2),axis=0))
print(fullfact([4,4])/3)

ff = fullfact([4,4])/3
ccd = np.unique(ccdesign(2),axis=0)
exp = np.unique(np.vstack((ff,ccd)),axis=0)
'''
lims = [[40,70],[20,50]]
c = build_rsm(200,lims)

cost_coef = c[0].coef_
beta_coef = c[1].coef_

m = optimize(c[0].coef_,c[1].coef_)
L = m.L.value*15+35
d = m.d.value*15+55
print("L = ",L)
print("d = ",d)
print("M = ",L+d)
print("Cost = ",m.obj())
print("beta = ",m.cons())
p = simulate_Periodic(200,111,44)
print(np.mean(p[0]))
print(np.mean(p[1]))

# Run 2

lims = [[40,70],[30,70]]
c = build_rsm(700,lims)

cost_coef = c[0].coef_
beta_coef = c[1].coef_

m = optimize(c[0].coef_,c[1].coef_)
L = m.L.value*15+35
d = m.d.value*15+55
print("L = ",L)
print("d = ",d)
print("M = ",L+d)
print("Cost = ",m.obj())
print("beta = ",m.cons())
p = simulate_Periodic(200,93,40)

# Run 3 

lims = [[30,50],[40,60]]
c = build_rsm(700,lims)

cost_coef = c[0].coef_
beta_coef = c[1].coef_

m = optimize(c[0].coef_,c[1].coef_)
L = m.L.value*15+35
d = m.d.value*15+55
print("L = ",L)
print("d = ",d)
print("M = ",L+d)
print("Cost = ",m.obj())
print("beta = ",m.cons())
p = simulate_Periodic(200,111,44)
print(np.mean(p[0]))
print(np.mean(p[1]))

# Run 4
lims = [[40,45],[40,45]]
c = build_rsm(700,lims)

cost_coef = c[0].coef_
beta_coef = c[1].coef_

m = optimize(c[0].coef_,c[1].coef_)
L = m.L.value*15+35
d = m.d.value*15+55
print("L = ",L)
print("d = ",d)
print("M = ",L+d)
print("Cost = ",m.obj())
print("beta = ",m.cons())
p = simulate_Periodic(200,111,44)
print(np.mean(p[0]))
print(np.mean(p[1]))

