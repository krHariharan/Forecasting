import pandas as pd
from datetime import *
import math

def power(factor: float):
    def p(t: float):
        t = t % 1.0
        return t ** factor
    return p

def partial_sin(factor: float, power: float, amplitude: float):
    def sin(t: float):
        t1 = t % 1.0
        t0 = t / amplitude
        return (t0 ** power) * math.sin(factor*t1)
    return sin

def partial_cos(factor: float, power: float, amplitude: float):
    def cos(t: float):
        t1 = t % 1.0
        t0 = t / amplitude
        return (t0 ** power) * math.cos(factor*t1)
    return cos

def beta_dist(a: float, b: float):
    denom = pow((a/(a+b)), a) * pow((b/(a+b)), b)
    def bd(t: float):
        t = t % 1.0
        return pow(t,a) * pow(1-t,b) / denom
    return bd

def sym_beta_dist(a: float, b: float, invert: int):
    denom = pow((a/(a+b)), a) * pow((b/(a+b)), b)
    def bd(t: float):
        t = t % 1.0
        if t < 0.5:
            t0 = 2*t
            return pow(t0,a) * pow(1-t0,b) / denom
        else:
            t0 = 2-2*t
            return pow(-1, invert) * pow(t0,a) * pow(1-t0,b) / denom
    return bd

def parameter_gen(start_date: datetime, end_date: datetime, freq: int):
    #Get dataframe with full date range, and normalize date range based on frequency
    d_range = pd.date_range(start_date, end_date)
    parameters = pd.DataFrame()
    parameters = parameters.reindex(d_range)
    parameters["t"] = ((parameters.index - start_date).days.astype('float64') / float(freq))
    amplitude = parameters["t"].max()
    
    #polynomial parameters
    parameters[str(freq)+"_t"] = parameters["t"].map(power(1.0))
    parameters[str(freq)+"_t^0.5"] = parameters["t"].map(power(0.5))
    parameters[str(freq)+"_t^2"] = parameters["t"].map(power(2))
    parameters[str(freq)+"_t^3"] = parameters["t"].map(power(3))
    parameters[str(freq)+"_t^4"] = parameters["t"].map(power(4))
    # trigonometric parameters
    for i in [0.0, 0.5, 1.0, 2.0, 3.0]:
        parameters[str(freq)+"_t^"+str(i)+"_sin"] = parameters["t"].map(partial_sin(2*math.pi, i, amplitude))
        parameters[str(freq)+"_t^"+str(i)+"_cos"] = parameters["t"].map(partial_sin(2*math.pi, i, amplitude))
        parameters[str(freq)+"_t^"+str(i)+"_half_sin"] = parameters["t"].map(partial_sin(math.pi, i, amplitude))
        parameters[str(freq)+"_t^"+str(i)+"_half_cos"] = parameters["t"].map(partial_sin(2*math.pi, i, amplitude))
        parameters[str(freq)+"_t^"+str(i)+"_quarter_sin"] = parameters["t"].map(partial_sin(math.pi/2, i, amplitude))
        parameters[str(freq)+"_t^"+str(i)+"_quarter_cos"] = parameters["t"].map(partial_sin(math.pi/2, i, amplitude))
    # beta distribution based
    for i in [0.5, 1, 2, 4, 8]:
        for j in [0.5, 1, 2, 4, 8]:
            parameters[str(freq)+"_bd_"+str(i)+"_"+str(j)] = parameters["t"].map(beta_dist(float(i), float(j)))
    # beta distribution based with symmetry
    for i in [0.5, 1, 2, 4, 8]:
        for j in [0.5, 1, 2, 4, 8]:
            if i != j:
                parameters[str(freq)+"_sym_bd_"+str(i)+"_"+str(j)] = parameters["t"].map(sym_beta_dist(float(i), float(j), 0))
                parameters[str(freq)+"_inv_bd_"+str(i)+"_"+str(j)] = parameters["t"].map(sym_beta_dist(float(i), float(j), 1))
    print(parameters)

if __debug__:
    parameter_gen(datetime(2020,1,1), datetime(2023,1,1), 364)