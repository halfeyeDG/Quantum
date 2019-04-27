import numpy as np
import itertools as it
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

C=np.array([[0.15617455, 0.06853744, 0.17302906, 0.1474065 , 0.01343537],\
            [0.1248934 , 0.14966344, 0.06479788, 0.05163321, 0.05804242],\
            [0.09472507, 0.07147226, 0.03362329, 0.19335372, 0.06208004],\
            [0.14649451, 0.08587618, 0.14655114, 0.08171785, 0.19225425],\
            [0.11847758, 0.06227386, 0.02067397, 0.05840247, 0.07507213]])

lam=0.5

K=3
N=len(C)

A=2*np.ones([N,N])-(1+2*K)*np.eye(N)
Q=np.triu(C+lam*A)

Qdict=dict()

for (i,j) in it.product(range(N),range(N)):
    if Q[i,j]!=0:
        Qdict.update({("x"+str(i),"x"+str(j)):Q[i,j]})

sampler = EmbeddingComposite(DWaveSampler())

response = sampler.sample_qubo(Qdict, num_reads=1000)
print(response)
