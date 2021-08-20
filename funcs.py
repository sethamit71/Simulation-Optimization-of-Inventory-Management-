# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:34:37 2021

@author: Utkarsh
"""
import numpy as np
from scipy.stats import t,norm
import pandas as pd
from pyDOE2 import ff2n, bbdesign, ccdesign, fullfact
from sklearn.linear_model import LinearRegression
from pyomo.environ import *
#np.random.seed(0)

def get_demand():
    u = np.random.uniform()
    if u <= 0.5:
        return 1
    elif u <= 3/4:
        return 2
    elif u <= 7/8:
        return 3
    else:
        return 4

def get_interDemand_time():
    return np.random.exponential(1/15)

def get_leadtime():
    return np.random.uniform(low=0.25,high=1.25)

def get_rush_leadtime():
    return np.random.uniform(low=0.1,high=0.25)

def get_halfwidth(data,conf=0.95):
    h = t.interval(conf,len(data)-1)[1]
    s = np.var(data,ddof=1)**0.5
    return h*s/len(data)**0.5

def get_reps(data,des_h,curr_h,conf=0.95):
    r1 = len(data)*curr_h**2/des_h**2
    z = norm.ppf((1+conf)/2)
    r2 = z*np.var(data,ddof=1)/des_h**2
    return[r1,r2]

def compare_alters(alt1,alt2,conf):
    assert len(alt1) == len(alt2)
    a = np.array(alt1)
    b = np.array(alt2)
    m = a-b
    xbar = np.mean(m)
    hw = get_halfwidth(m,conf)
    r = [xbar-hw,xbar+hw]
    if xbar-hw <= 0 and xbar+hw >= 0:
        return r,0
    elif xbar-hw > 0:
        return r,1
    elif xbar+hw < 0:
        return r,2
    else:
        print("Unthinkable has happened")
        print(r,xbar-hw,xbar+hw)
        return r,-1

def simulate_Periodic(reps,M,L):
    means = []
    frs = []
    for i in range(reps):
        month = 1
        td = get_interDemand_time()
        to = np.inf
        to1 = np.inf
        inv = 50
        b = 0
        cost = 0
        N = 112
        order = 0
        order1 = 0
        costs = []
        tot_demand = 0
        unsat_demand = 0 
        fill_rates = []
        while month <= N:
            if min(month,td,to,to1) == month:
                te = month
                month = month+1
                I = inv + order + order1
                if I < L and inv > 0:
                    l = get_leadtime()
                    if order == 0:
                        to = te + l
                        order = M-I
                    else:
                        to1 = te + l
                        order1 = M-I
                    cost = cost+(M-I)*5+60
                if inv <= 0:
                    l = get_rush_leadtime()
                    if order == 0:
                        to = te + l
                        order = M-I
                    else:
                        to1 = te + l
                        order1 = M-I
                    cost = cost + (M-I)*12+120
                costs.append(cost+max(inv,0) + 4*b)
                if month > 12:
                    fill_rates.append(1-unsat_demand/tot_demand)
                unsat_demand = 0
                tot_demand = 0
                cost = 0
                b = 0
            elif min(month,to,td,to1) == to:
                te = to
                inv = inv + order
                order = 0
                to = np.inf
            elif min(month,to,td,to1) == to1:
                te = to1
                inv = inv + order1
                order1 = 0
                to1 = np.inf
            elif min(month,to,td,to1) == td:
                te = td
                td = td + get_interDemand_time()
                d = get_demand()
                tot_demand += d
                if inv <= 0:
                    b += d
                    unsat_demand += d
                elif inv-d < 0:
                    b += d-inv
                    unsat_demand += d-inv
                inv = inv - d
        c = np.array(costs[12:])
        means.append(np.mean(c))
        frs.append(np.mean(fill_rates))
    return means,frs

def simulate_Continuous(reps,M,L):
    cost_data = []
    fill_rate_data = []
    for r in range(reps):
        orders_T = []
        orders_Q = []
        tc = get_interDemand_time()
        to = np.inf
        q = 0
        te = tc
        b = 0
        inv = 50
        ind = 0
        I = 0
        debug_data = pd.DataFrame(data=None,columns=['Te','Inv','EType','lenT','lenQ','Demand/Q','orderType','delivery_time','I','b','Ordering_cost'],index=range(5000))
        while tc <= 112:
            if min(tc,to) == tc:
                te = tc
                d = get_demand()
                debug_data.at[ind,'b'] = 0
                if inv-d <= 0 and inv > 0:
                    b += d-inv
                    debug_data.at[ind,'b'] = d-inv
                elif inv <= 0:
                    b += d
                    debug_data.at[ind,'b'] = d
                inv = inv-d
                I = inv+sum(i for i in orders_Q)
                debug_data.loc[ind,'orderType'] = '-1'
                debug_data.loc[ind,'delivery_time'] = -1
                debug_data.loc[ind,'Ordering_cost'] = 0
                if I <= L-1 and inv > 0:
                    x = te+get_leadtime()
                    orders_T.append(x)
                    orders_Q.append(M-I)
                    minpos = orders_T.index(min(orders_T))
                    to = orders_T[minpos]
                    q = orders_Q[minpos]
                    debug_data.loc[ind,'orderType'] = 'N'
                    debug_data.loc[ind,'delivery_time'] = x
                    debug_data.loc[ind,'Ordering_cost'] = 90+5*(M-I)
                    # add cost
                elif inv <= 0:
                    x = te+get_rush_leadtime()
                    orders_T.append(x)
                    orders_Q.append(M-I)
                    minpos = orders_T.index(min(orders_T))
                    to = orders_T[minpos]
                    q = orders_Q[minpos]
                    debug_data.loc[ind,'orderType'] = 'R'
                    debug_data.loc[ind,'delivery_time'] = x
                    debug_data.loc[ind,'Ordering_cost'] = 120+12*(M-I)
                    # add cost
                tc = te + get_interDemand_time()
                debug_data.loc[ind,'Te'] = te
                debug_data.loc[ind,'Inv'] = inv
                debug_data.loc[ind,'EType'] = 'C'
                debug_data.loc[ind,'lenT'] = len(orders_T)
                debug_data.loc[ind,'lenQ'] = len(orders_Q)
                debug_data.loc[ind,'Demand/Q'] = d
                debug_data.loc[ind,'I'] = I
                ind += 1
            if min(tc,to) == to:
                te = to
                inv = inv + q
                orders_T.remove(to)
                orders_Q.remove(q)
                debug_data.loc[ind,'Te'] = te
                debug_data.loc[ind,'Inv'] = inv
                debug_data.loc[ind,'EType'] = 'D'
                debug_data.loc[ind,'lenT'] = len(orders_T)
                debug_data.loc[ind,'lenQ'] = len(orders_Q)
                debug_data.loc[ind,'Demand/Q'] = q
                debug_data.loc[ind,'orderType'] = '-1'
                debug_data.loc[ind,'delivery_time'] = -1
                debug_data.loc[ind,'I'] = I
                debug_data.at[ind,'b'] = 0
                if len(orders_T) == 0:
                    to = np.inf
                    q = 0
                else:
                    minpos = orders_T.index(min(orders_T))
                    to = orders_T[minpos]
                    q = orders_Q[minpos]
                ind += 1
        
        debug_data = debug_data.dropna()
        
        debug_data['Month'] = debug_data['Te'].astype(int)+1
        debug_data['shifted_Te'] = debug_data['Te'].shift(1,fill_value=0.0)
        debug_data['deltaT'] = -(debug_data['shifted_Te']-debug_data['Te'])
        debug_data['Holding_cost'] = debug_data['deltaT']*debug_data['Inv']
        ind2 = debug_data.index[debug_data['Holding_cost']<0]
        for k in ind2:
            debug_data.at[k,'Holding_cost'] = 0
        #debug_data['Holding_cost'][ind2] = 0
        debug_data['BackOrder_Cost'] = debug_data['b']*4 
        debug_data['Total_cost']= debug_data['Ordering_cost']+debug_data['Holding_cost']+debug_data['BackOrder_Cost']
        ss = debug_data[['Total_cost','b','Demand/Q','EType']].groupby(debug_data['Month']).sum()
        #debug_data['level'] = debug_data['Inv']+debug_data['Demand/Q']
        c = ss['Total_cost'][12:].mean()
        fr_data = debug_data[(debug_data['EType']=='C')&(debug_data['Month']>=13)]
        fill_rate = 1-fr_data['b'].sum()/fr_data['Demand/Q'].sum()
        cost_data.append(c)
        fill_rate_data.append(fill_rate*100)
    return cost_data,fill_rate_data

def perform_DOE(reps):
    d = ff2n(3).tolist()
    res = {}
    exp = 0
    exps = []
    for i in d:
        M = 75+i[1]*25
        L = 35+i[2]*5 
        if int(i[0]) == 1:
            res[exp] = [simulate_Periodic(reps, M, L)]
        elif int(i[0]) == -1:
            res[exp] = [simulate_Continuous(reps, M, L)]
        exp += 1
        print("  Completed Experiment",exp)
    return res,d

def get_rsm_doe():
    ff = fullfact([5,5])/4*2-1
    ccd = np.unique(ccdesign(2),axis=0)
    exp = np.unique(np.vstack((ff,ccd)),axis=0)
    return exp

def get_rsm_data(exp,rM,rL,reps,s):
    d = exp[0]*(rM[1]-rM[0])/2+(rM[1]+rM[0])/2
    L = exp[1]*(rL[1]-rL[0])/2+(rL[1]+rL[0])/2
    M = L+d
    if s == 'P':
        result = simulate_Periodic(reps, M, L)
        return result
    elif s == 'C':
        result = simulate_Continuous(reps, M, L)
        return result
    else:
        print("Invalid value of s. s=",s)
        return -1
    

def gather_data(exps,lims,reps,s):
    costs = []
    betas = []
    for i in exps:
        r = get_rsm_data(i,lims[0],lims[1],reps,s)
        if type(r) == int:
            return
        costs.append(np.mean(r[0]))
        betas.append(np.mean(r[1]))
    return costs,betas

def get_features(exp):
    e = pd.DataFrame(data=exp)
    e['i'] = 1
    e['mult'] = e[0]*e[1]
    e['s1'] = e[0]**2
    e['s2'] = e[1]**2
    e = e[['i',0,1,'mult','s1','s2']]
    return e.to_numpy()

def build_rsm(reps,lims):
    e = get_rsm_doe()
    d = gather_data(e,lims,reps,'P')
    f = get_features(e)
    cost = LinearRegression(fit_intercept=False)
    cost.fit(f,d[0])
    beta = LinearRegression(fit_intercept=False)
    beta.fit(f,d[1])
    print("Cost score :",cost.score(f,d[0]))
    print("Beta score :",beta.score(f,d[1]))
    return cost, beta

def optimize(cost_coef,beta_coef):
    m = ConcreteModel()
    m.L = Var(domain=Reals)
    m.d = Var(domain=Reals)
    m.pprint()
    m.cons = Constraint(expr=beta_coef[0]+m.d*beta_coef[1]+m.L*beta_coef[2]+m.d*m.L*beta_coef[3]+m.d**2*beta_coef[4]+m.L**2*beta_coef[5] >= 98.7)
    m.obj = Objective(expr=cost_coef[0]+m.d*cost_coef[1]+m.L*cost_coef[2]+m.d*m.L*cost_coef[3]+m.d**2*cost_coef[4]+m.L**2*cost_coef[5])
    opt = SolverFactory('couenne',executable='D:\IITB\secondSem\IE630\project\couenne.exe')
    result = opt.solve(m,tee=False)
    print("Solver status:",result.solver.status)
    print("Termination condition :", result.solver.termination_condition)
    return m
    
    
    
    
    