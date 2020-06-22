import torch
import torch.nn as nn
from torch.autograd import Variable, grad, Function
import math
import numpy as np
from numba import jit
from itertools import product


def dtw(D):
    ######### Given a distance matrix D_xy, computes DTW and the optimal alignment matrix A
    ### Input: D, dim: T_x x T_y
    ### Output A, dim: T_x x T_y
    
        m=D.shape[0]
        n=D.shape[1]

        R=torch.zeros((m+1,n+1))
        R[1:,0]=np.inf
        R[0,1:]=np.inf

        ### Forward recursion to compute DTW. The value of DTW correspongs to R[-1,-1] ###
        for i in range(1,m+1):
            for j in range(1,n+1):
                R[i,j]=D[i-1,j-1]+np.min((R[i,j-1],R[i-1,j],R[i-1,j-1]))

        A=torch.zeros((m,n))
        k,l = m-1,n-1
        A[0,0]=A[k,l]=1
        
               ### Backward recursion to compute the optimal alignment matrix A ###
        while (k>=0 and l>=0):
            
            move = np.argmin((R[k,l+1],R[k+1,l],R[k,l]))
            if move==2:
                A[k-1,l-1]=1
                k=k-1
                l=l-1
            elif move==0:
                A[k-1,l]=1
                k=k-1

            elif move==1:
                A[k,l-1]=1
                l=l-1


        return(A)
    
    
##   
##  The code for soft-DTW is adapted from https://github.com/Sleepwalking/pytorch-softdtw
##    
    
    
@jit(nopython = True)
def compute_softdtw(D, gamma):
        ######### Given a distance matrix D_xy, and entropic coefficient \gamma, computes soft-DTW forward
        ### Input: D, dim: n_batches x T_x x T_y
        ###        gamma, dim: 1
        ### Output R, dim: n_batches x T_x x T_y
        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        R = np.ones((B, N + 2, M + 2)) * np.inf
        R[:, 0, 0] = 0
        for k in range(B):
            for j in range(1, M + 1):
                for i in range(1, N + 1):
                    r0 = -R[k, i - 1, j - 1] / gamma
                    r1 = -R[k, i - 1, j] / gamma
                    r2 = -R[k, i, j - 1] / gamma
                    rmax = max(max(r0, r1), r2)
                    rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                    softmin = - gamma * (np.log(rsum) + rmax)
                    R[k, i, j] = D[k, i - 1, j - 1] + softmin
        return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
    ######### Given a distance matrix D_xy, forward values in the forward recursion of DTW and entropic coefficient \gamma, performs a backward pass, which computes the optimal soft-alignment matrix A
    
        ### Input: D, dim: n_batches x T_x x T_y
        ###        R, dim: n_batches x T_x x T_y
        ###        gamma, dim: 1
        ### Output A, dim: n_batches x T_x x T_y
        
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, : , -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


class _SoftDTW(Function):
    
        @staticmethod
        def forward(ctx, D, gamma):
            ######### Forward pass computing soft-DTW given a distance matrix D and an entropic coefficient gamma
    
        ### Input: D, dim: n_batches x T_x x T_y
        ###        gamma, dim: 1
        ### Output R, dim: n_batches x T_x x T_y

                gamma = torch.Tensor([gamma]) # dtype fixed
                D_ = D.detach().cpu().numpy()
                g_ = gamma.item()
                R=compute_softdtw(D_, g_)
                R = torch.tensor(R, dtype=torch.double)
                ctx.save_for_backward(D, R, gamma)
                return R[:, -2, -2]

        @staticmethod
        def backward(ctx, grad_output):
             ######### Backward pass computing the soft-alignment matrix

        ### Output E, dim: n_batches x T_x x T_y
                dev = grad_output.device
                dtype = grad_output.dtype
                D, R, gamma = ctx.saved_tensors
                D_ = D.detach().cpu().numpy()
                R_ = R.detach().cpu().numpy()
                g_ = gamma.item()
                E = torch.Tensor(compute_softdtw_backward(D_, R_, g_))
                return E, None
