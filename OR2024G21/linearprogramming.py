from scipy import optimize as op
import numpy as np
c=np.array([0.19208271,0.169279,0.02415765,0.40700246,0.20747819])
c=c/0.5952913619503445
A_ub=np.array([[0,0,-1,0,0]])
B_ub=np.array([-3])
A_eq=np.array([[7000,-5000,20000,100000,10000]])
B_eq=np.array([500000])
x1=(0,50)
x2=(0,30)
x3=(0,4.5)
x4=(0,2)
x5=(0,2)
res=op.linprog(-c,A_ub,B_ub,A_eq,B_eq,bounds=(x1,x2,x3,x4,x5))
print(res)
