import numpy as np
N = 10
A = np.concatenate([np.zeros(int(N/2)), np.ones(int(N/2))])
R = np.zeros([N,5])
for i in range(5):
    R[:,i] = np.random.permutation(np.arange(N))
    
    
x = np.random.normal(0,1,N)

# Add a column of ones
X = np.c_[np.ones(x.shape[0]), x]
Q,R = np.linalg.qr(X)
R\(np.dot(Q,x))
# x2fx(design);
    # [Q,R] = qr(design,0);
    # beta = R\(Q'*y);