import numpy as np
import itertools as it

C=np.array([[0.15617455, 0.06853744, 0.17302906, 0.1474065 , 0.01343537],\
            [0.1248934 , 0.14966344, 0.06479788, 0.05163321, 0.05804242],\
            [0.09472507, 0.07147226, 0.03362329, 0.19335372, 0.06208004],\
            [0.14649451, 0.08587618, 0.14655114, 0.08171785, 0.19225425],\
            [0.11847758, 0.06227386, 0.02067397, 0.05840247, 0.07507213]])


w=np.ones([1,5])
def Port_var(w,C):
    return np.dot(w,np.dot(C,w.T))[0][0]

b=[0,1]

Hs=[]
ws=[]

for w in it.product(b,b,b,b,b):
    if sum(w)==3:
        Hs.append(Port_var(np.array([w]),C))
        ws.append(w)

print(ws[np.argmin(Hs)])
