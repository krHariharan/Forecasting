import pandas as pd
from datetime import *
import math

def linear(t):
    return 2*t-1

def partial_sin(factor: float):
    def sin(t: float):
        return math.sin(factor*t)
    return sin

def partial_cos(factor: float):
    def cos(t: float):
        return math.cos(factor*t)
    return cos

def beta_dist(a: float, b: float):
    denom = pow((a/(a+b)), a) * pow((b/(a+b)), b)
    def bd(t: float):
        return pow(t,a) * pow(1-t,b) / denom
    return bd

def sym_beta_dist(a: float, b: float, invert: int):
    denom = pow((a/(a+b)), a) * pow((b/(a+b)), b)
    def bd(t: float):
        if t < 0.5:
            t0 = 2*t
            return pow(t0,a) * pow(1-t0,b) / denom
        else:
            t0 = 2-2*t
            return pow(-1, invert) * pow(t0,a) * pow(1-t0,b) / denom
    return bd

def parameter_gen(start_date: datetime, end_date: datetime, freq: int):
    d_range = pd.date_range(start_date, end_date)
    parameters = pd.DataFrame()
    parameters = parameters.reindex(d_range)
    parameters["t"] = ((parameters.index - start_date).days.astype('float64') / float(freq)) % 1.0
    
    parameters[str(freq)+"_linear"] = parameters["t"].map(linear)
    parameters[str(freq)+"_sin"] = parameters["t"].map(partial_sin(2*math.pi))
    parameters[str(freq)+"_cos"] = parameters["t"].map(partial_sin(2*math.pi))
    parameters[str(freq)+"_half_sin"] = parameters["t"].map(partial_sin(math.pi))
    parameters[str(freq)+"_half_cos"] = parameters["t"].map(partial_sin(2*math.pi))
    parameters[str(freq)+"_quarter_sin"] = parameters["t"].map(partial_sin(math.pi/2))
    parameters[str(freq)+"_quarter_cos"] = parameters["t"].map(partial_sin(math.pi/2))
    for i in [0.5, 1, 2, 4, 8]:
        for j in [0.5, 1, 2, 4, 8]:
            parameters[str(freq)+"_bd_"+str(i)+"_"+str(j)] = parameters["t"].map(beta_dist(float(i), float(j)))
    for i in [0.5, 1, 2, 4, 8]:
        for j in [0.5, 1, 2, 4, 8]:
            if i != j:
                parameters[str(freq)+"_sym_bd_"+str(i)+"_"+str(j)] = parameters["t"].map(sym_beta_dist(float(i), float(j), 0))
                parameters[str(freq)+"_inv_bd_"+str(i)+"_"+str(j)] = parameters["t"].map(sym_beta_dist(float(i), float(j), 1))
    print(parameters)

parameter_gen(datetime(2020,1,1), datetime(2023,1,1), 364)
