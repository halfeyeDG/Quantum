import numpy as np
import itertools as it

r=np.array([[0.01,0.02,0.03,0.04,0.05]])


C=np.array([[ 0.01  , -0.0134,  0.0976,  0.0458, -0.0192],\
            [ 0.    ,  0.0225,  0.0291,  0.0958,  0.0264],\
            [ 0.    ,  0.    ,  0.04  ,  0.0728,  0.0098],\
            [ 0.    ,  0.    ,  0.    ,  0.0625,  0.0094],\
            [ 0.    ,  0.    ,  0.    ,  0.    ,  0.09  ]])

gamma=0.6



w=np.ones([1,5])
def obj(r,w,C,gamma):
    return (-np.dot(r,w.T)+gamma*2*np.dot(w,np.dot(C,w.T)))[0][0]

b=[0,1]

Hs=[]
ws=[]

for w in it.product(b,b,b,b,b):
    if sum(w)==3:
        Hs.append(obj(r,np.array([w]),C,gamma))
        ws.append(w)

print(ws[np.argmin(Hs)])
