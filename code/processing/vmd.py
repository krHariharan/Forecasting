#%% Simple example: generate signal with 3 components + noise  
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD 
import pandas as pd 

import sys
inputFile = ""
if len(sys.argv) > 1:
    inputFile = sys.argv[1]
else:
    print("Enter file to process")
    inputFile = input()

inputData = pd.read_csv("../../data/raw/"+inputFile, thousands=",")[["Date", "Price"]]

#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 20              # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7  


#. Run VMD 
u, u_hat, omega = VMD(inputData["Price"], alpha, tau, K, DC, init, tol)  

print(u)
for dataset in u:
    print(len(dataset))

#. Visualize decomposed modes
plt.figure()
plt.subplot(2,1,1)
plt.plot(inputData["Price"])
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(u.T)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
# plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.tight_layout()
plt.savefig("vmd.png")
