import torch
import torch.nn as nn
from torch.autograd import Variable, grad, Function

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

from itertools import product

from .DTW import dtw, _SoftDTW


class gromov_dtw(nn.Module):
    def __init__(self, max_iter, gamma = 0.5, loss_only = 0, device = 'cpu',
                 dtw_approach = 'GDTW', verbose = 1, tol = 1e-3):
        
        super(gromov_dtw, self).__init__()

        self.max_iter = max_iter
        self.device = device
        self.gamma = gamma
        self.dtw_approach = dtw_approach
        self.verbose = verbose
        self.loss_only = loss_only
        self.losses_gdtw = []
        self.tol=tol


    def forward(self, x, y, xteps=1e-3):
        #### Computes GDTW and returns the optimal alignment matrix and optimal path
        # 
        
        #Computes cost matrices C_x and C_y
        C1 = self._cost_matrix(x, x)
        C2 = self._cost_matrix(y, y)

        if self.loss_only:
                
                loss = self.g_dtw(C1,C2)
                return loss
        else:
                loss, A = self.g_dtw(C1, C2)

        return( loss, A )

    
    def init_alignment(self,C1,C2):
        ###  Initializes the alignment matrix to the identity plus extra ones on the sides to make it a proper alignment
        
        A = torch.eye(C1.shape[0], C2.shape[0]).float()
        
        ### If alignment is rectangular - add ones to make it a proper alignment along the largest dimension
        if C1.shape[0] > C2.shape[0]:
            A[C2.shape[0]:,-1] = 1
            
        elif C2.shape[0] > C1.shape[0]:
            A[-1,C1.shape[0]:] = 1
        
        return(A)
    
    def _cost_matrix(self, x, y, p=2):
        ###  Computes a cost matrix given two tensors

        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)


        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

        return C.float()

    def compute_constc(self, f1C1, f2C2, A):
        ###  Computes terms in the tensor contraction given the cost matrices and the current alignment

        p = torch.sum(A, axis=1).reshape(-1, 1)
        q = torch.sum(A, axis=0).reshape(-1, 1)

        ones_m = torch.ones(len(p)).float().reshape(-1,1)
        ones_n = torch.ones(len(q)).float().reshape(-1,1)

        constC1 = torch.matmul(torch.matmul(f1C1, p), ones_n.T)
        constC2 = torch.matmul(ones_m, torch.matmul(q.T, f2C2.T))

        constC = constC1 + constC2
        
        return(constC)


    def tensor_product(self, f1C1, f2C2, hC1, hC2, A):
        ###  Computes the tensor contraction   $L \otimes A$    given the current alignment 
        
        constC = self.compute_constc(f1C1, f2C2, A)
        
        AB = - torch.sparse.mm(A.T.to_sparse(), hC1.T).T.matmul(hC2.T)
        
        tens = constC + AB
        
        return tens
    
        
    def compute_dtw_alignment(self, tens):
        ###  Computes the DTW alignment for the cost $ D = L \otimes A $
        
        if self.dtw_approach == 'GDTW':
                A = dtw(tens)

        elif self.dtw_approach == 'soft_GDTW':
                if not tens.requires_grad:
                    tens = torch.autograd.Variable(tens, requires_grad = True)
                    
                tens = tens.reshape(1, tens.shape[0], tens.shape[1])

                sdtw_pyt = _SoftDTW()
                
                loss = sdtw_pyt.apply(tens, self.gamma)

                A = torch.autograd.grad(loss, tens, create_graph=True)[0][0]   
                
        
        return(A)
    
    def g_dtw(self,C1, C2):
        ### Computes GDTW, the optimal alignment matrix and optimal path
        
        #Initialize the alignment matrix A
        A = self.init_alignment(C1, C2)
        
        #Compute terms of the tensor contraction
        hC1, hC2 = C1, 2*C2
        f1C1, f2C2 = C1**2, C2**2

        cpt = 0
        err=1

        while (err > self.tol and cpt < self.max_iter):
    
            # Compute tensor contraction $ tens = L \otimes A $
            tens = self.tensor_product(f1C1, f2C2, hC1, hC2, A)
            
            # Computes $ < L \otimes A, A > $, i.e. current value of the GDTW functional 
            self.losses_gdtw.append( torch.sum(tens * A) )
            self.A = A
            # Perform linear minimization oracle step  $\argmin_{A'} < L \otimes A, A' > $
            A = self.compute_dtw_alignment(tens)

            
            if cpt>0:
                err = torch.abs(self.losses_gdtw[-2] - self.losses_gdtw[-1])
                
            if self.verbose:
                print('iter:', cpt, 'GDTW:', self.losses_gdtw[-1].item())
                
                if err<0:
                    print('FW step-size = 0')
            
        

            cpt += 1
   

        if self.loss_only:
                return self.losses_gdtw[-2]
        else:
                return self.losses_gdtw[-2], self.A


