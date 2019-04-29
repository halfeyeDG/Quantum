import numpy as np
import itertools as it
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

#リターンベクトル
r=np.array([0.01,0.02,0.03,0.04,0.05])

#共分散行列(下三角部分は省略)
C=np.array([[ 0.01  , -0.0134,  0.0976,  0.0458, -0.0192],\
            [ 0.    ,  0.0225,  0.0291,  0.0958,  0.0264],\
            [ 0.    ,  0.    ,  0.04  ,  0.0728,  0.0098],\
            [ 0.    ,  0.    ,  0.    ,  0.0625,  0.0094],\
            [ 0.    ,  0.    ,  0.    ,  0.    ,  0.09  ]])

#N=5資産中K=3資産の等ウェイトポートフォリオ
N=len(C)
K=3

#ガンマとラムダを定義
gamma=0.6
lam=0.5


#QUBO matrix を計算
R=np.diag(r)
A=2*np.ones([N,N])-(1+2*K)*np.eye(N)

Q=-R+gamma*C+lam*A

#問題を与える時は辞書にして渡す必要があるので変換

Qdict=dict()

for i in range(N):
    for j in range(N):
        if i<=j:
            Qdict.update({("x"+str(i),"x"+str(j)):Q[i,j]})


sampler = EmbeddingComposite(DWaveSampler())

#問題をapiに投げ、結果を取得
response = sampler.sample_qubo(Qdict,num_reads=100)
print(response)
