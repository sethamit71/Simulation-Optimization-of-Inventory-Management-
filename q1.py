# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:43:14 2021

@author: Utkarsh
"""

from funcs import *
import pandas as pd
'''
reps = 10
M = 100
L = 40
rep10,fr10 = simulate_Periodic(reps,M,L)
m10 = np.mean(rep10)
h10 = get_halfwidth(rep10)
fill_rate = np.mean(fr10)
fr_hw10 = get_halfwidth(fr10)

rep_new = int(max(get_reps(rep10,5,h10)))
rep_data_new,fr_new = simulate_Periodic(rep_new,M,L)
m_new = np.mean(rep_data_new)
h_new = get_halfwidth(rep_data_new)
fill_rate_new = np.mean(fr_new)
fr_hw_new = get_halfwidth(fr_new)

print("10 reps: mean",m10,"Half width",h10,"Fill rate",fill_rate,"FRhw",fr_hw10)
print(rep_new,"reps: mean",m_new,"Half width",h_new,"Fill rate",fill_rate_new,"FRhw",fr_hw_new)


alt_c1, alt_fr1 = simulate_Periodic(100, 100, 40)
alt_c2, alt_fr2 = simulate_Periodic(100, 50, 30)

print(compare_alters(alt_c1, alt_c2, 0.95))
print(compare_alters(alt_fr1, alt_fr2, 0.95))
'''

#doe_result = perform_DOE(1000)

c = 0.1/28.0
cost_comp = pd.DataFrame(data=None,columns=['exp1','exp2','range'],index=range(int((len(doe_result[0])-1)*(len(doe_result[0]))/2)))
ind = 0
for i in range(len(doe_result[1])-1):
    for j in range(i+1,len(doe_result[1])):
        res_exp1 = doe_result[0][i][0][0]
        res_exp2 = doe_result[0][j][0][0]
        r,_ = compare_alters(res_exp1,res_exp2,c)
        cost_comp.at[ind,'exp1'] = doe_result[1][i]
        cost_comp.at[ind,'exp2'] = doe_result[1][j]
        cost_comp.at[ind,'range'] = r
        ind += 1
        
cost_comp.to_csv('Comparisons.csv')
