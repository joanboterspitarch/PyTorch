import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import torch.nn.functional as F

def Q(theta):
    """ determines the quadrant of the angle
    Args:
        theta (float): angle between 0 and 2pi
    Returns:
        int: number of the quadrant
    """
    if 0 <= theta < np.pi/2 :
        return 1
    elif np.pi/2 <= theta < np.pi :
        return 2
    elif np.pi <= theta < (3/2)*np.pi :
        return 3
    elif (3/2)*np.pi <= theta <= 2*np.pi :
        return 4

def T(theta):
    """This function transports any angle to the first quadrant
    Args:
        theta (float): angle
    Returns:
        float: theta-like angle in first quadrant
    """
    if Q(theta) == 1 :
        return theta
    elif Q(theta) == 2 :
        return np.pi - theta    
    elif Q(theta) == 3 :
        return theta - np.pi
    elif Q(theta) == 4 :
        return 2*np.pi - theta

def FunDiag(xi):
    """determines the probability of the diagonal elements
    Args:
        theta (float): angle
    Returns:
        float: infection probability for diagonal elements which are in the first layer
    """
    
    if 0 < xi <= torch.pi/4 :
        return torch.tan(xi)
    else:
        return 1/torch.tan(xi)

# def RotacionIzquierda(array):
#    """ rotates to the left of our array
#    Args:
#        array (array): array which will be rotated
#    Returns:
#        array: we get our rotated array
#    """
#    return torch.rot90(array, k=1)
#    new_array = torch.t(array)
#    return new_array

#def RotacionDerecha(array):
#    """ rotates to the right of our array
#    Args:
#        array (array): array which will be rotated
#    Returns:
#        array: we get our rotated array
#    """
#    new_array = torch.rot90(array, k=-1)
#    new_array = torch.t(array)
#    return new_array

#RotacionDerecha = torch.rot90

#RotacionDerecha(array, k=-1) == torch.rot90(array, k=-1)

def Rotacion(array, theta):
    """ this function will rotate our array depending on theta
    Args:
        array (array): array which will be rotated
        theta (float): angle
    Returns:
        array: we get our rotated array
    """
    if Q(theta)==1:
        return array
    elif Q(theta)==2:
        return torch.t(torch.rot90(array, k=1))
    elif Q(theta)==3:
        return torch.rot90(array, k=2)
    elif Q(theta)==4:
        return torch.t(torch.rot90(array, k=-1))

def init (N, K):
    """Initialize our arrays in order to start the model.
    Args:
        N (int): size of the grid.
        K (int): number of generations.
    Returns:
        array : our storage of our grid.
        array : our storage of our probabilities matrix. 
    """
    E = torch.zeros(N,N,K+1, dtype=torch.int8)
    P = torch.zeros(N,N,K, dtype=torch.float64)
    return E, P


class Grid:

    def __init__ (self, N, K):
        self.N = N
        self.K=K
        self.n = int(N/2)
        self.state = torch.zeros(self.N, self.N, dtype = torch.int8)
        self.state[self.n, self.n] = 1
        self.ind = [(self.n, self.n)]
        self.cont = self.state.clone()
        self.susceptibles = self.N**2 - 1
        self.infecteds = 1
        self.deads = 0
        self.neigh_prob = torch.zeros(self.N, self.N, dtype = torch.float64)
        self.columnas = ['Theta', 'Quadrant', 'Xi', 'Rho', 'm', 'Susceptibles', 'Infecteds', 'Deads']
        self.df = pd.DataFrame(index = range(self.K + 1), columns = self.columnas)
        self.df.iloc[0] = [0, 0, 0, 0, 0, self.susceptibles, self.infecteds, self.deads]
    
    def __param__(self, inc=1, partition=[0.1, 0.5, 0.9], p0=0.25, div = 2): 
        self.inc = inc
        self.partition = partition
        self.div = div
        self.p0 = p0

    def submatrix(self):
        
        sin = torch.sin(self.xi)
        cos = torch.cos(self.xi)
        f = torch.where(
            self.xi <= torch.pi/4,
            torch.tan(self.xi),
            1/torch.tan(self.xi)
        )

        self.A = torch.zeros((3, 3, self.K), dtype = torch.float64)
        self.A[0, 1, :] = sin
        self.A[0, 2, :] = f
        self.A[1, 2, :] = cos

    def enlargement_process(self):

        #self.size_large_matrix = torch.where(
        #    self.m <= 1,
        #    3,
        #    (3-self.m)*5 + (self.m-2)*7
        #)

        #self.exp = torch.where(
        #    self.m <= 1,
        #    1,
        #    (3-self.m)*2 + (self.m-2)*3
        #)

        # creating matrices
        
        ## 1. When is between u_2, u_3
        M2 = torch.zeros(size=(5, 5, self.K), dtype=torch.float64)
        M2[1:-1, 1:-1, :] = self.A
        M2[0, 2, :] = M2[1, 2, :]/self.div
        M2[0, 4, :] = M2[1, 3, :]/self.div
        M2[2, 4, :] = M2[2, 3, :]/self.div
        M2[0, 3, :] = (M2[0, 2, :] + M2[0, 4, :])/2
        M2[1, 4, :] = (M2[0, 4, :] + M2[2, 4, :])/2

        ## 2. When is between u_3, 1
        M3 = torch.zeros(size=(7, 7, self.K), dtype=torch.float64)
        M3[1:-1, 1:-1, :] = M2
        M3[0, 3, :] = M3[1, 3, :]/self.div
        M3[0, 4, :] = M3[1, 4, :]/self.div
        M3[0, 6, :] = M3[1, 5, :]/self.div
        M3[0, 5, :] = (M3[0, 4, :] + M3[0, 6, :])/2
        M3[2, 6, :] = M3[2, 5, :]/self.div
        M3[3, 6, :] = M3[3, 5, :]/self.div
        M3[1, 6, :] = (M3[0, 6, :] + M3[2, 6, :])/2

        # Final tensor after enlargement process
        self.large_matrix = torch.zeros(size=(7, 7, self.K), dtype=torch.float64)

        self.large_matrix[2:5, 2:5, self.m==0] = self.p0
        self.large_matrix[3, 3, self.m==0] = 0
        self.large_matrix[2:5, 2:5, self.m==1] = self.A[:, :, self.m==1]
        self.large_matrix[1:-1, 1:-1, self.m==2] = M2[:, :, self.m==2]
        self.large_matrix[:, :, self.m==3] = M3[:, :, self.m==3]
        
        self.large_matrix[:, :, self.q == 2] = torch.transpose(
            torch.rot90(
                self.large_matrix[:, :, self.q == 2],
                k=1
            ),
            0,
            1
        )

        self.large_matrix[:, :, self.q == 3] = torch.rot90(
            self.large_matrix[:, :, self.q == 3],
            k = 2
        )

        self.large_matrix[:, :, self.q == 4] = torch.transpose(
            torch.rot90(
                self.large_matrix[:, :, self.q == 4],
                k=-1
            ),
            0,
            1
        )
        

    def neighbourhood_relation(self, step):
        padding = torch.zeros(self.N + 6, self.N + 6, dtype=torch.float64)
        for i,j in self.ind:
            padding[i:(i+7), j:(j+7)] += self.large_matrix[:, :, step]
        self.neigh_prob = padding[3:-3, 3:-3].clone()

    def update(self, tau=1):
    
        #self.state = torch.where(
        #    self.cont==self.inc,
        #    2,
        #    self.state
        #)
        self.state[self.cont==self.inc] = 2

        #actualizamos los estados fáciles
        #self.state = torch.where(
        #    torch.logical_and(self.neigh_prob >= 1, self.state==0),
        #    1,
        #    self.state
        #)
        self.state[torch.logical_and(self.neigh_prob >= 1, self.state==0)] = 1

        #self.state = torch.where(
        #    torch.logical_and(self.neigh_prob <= 0, self.state==0),
        #    0,
        #    self.state
        #)
        self.state[torch.logical_and(self.neigh_prob <= 0, self.state==0)] = 0

        logits = torch.stack([1 - self.neigh_prob, self.neigh_prob], dim=-1).flatten().view((self.N, self.N, 2)).log()
        #self.state[torch.logical_and(self.neigh_prob >= 1, self.state==0)] = 1
        #self.state[torch.logical_and(self.neigh_prob <= 0, self.state==0)] = 0

        #actualizamos probabilidades entre 0 y 1
        pos = torch.logical_and(self.state==0, torch.logical_and(self.neigh_prob < 1, self.neigh_prob > 0))
        upd = F.gumbel_softmax(logits=logits, tau=tau, hard=True).index_select(dim=2, index=torch.tensor([1])).squeeze()
        #self.state = torch.where(
        #    torch.logical_and(self.state==0, torch.logical_and(self.neigh_prob < 1, self.neigh_prob > 0)),
        #    F.gumbel_softmax(logits=logits, tau=tau, hard=True).index_select(dim=2, index=torch.tensor([1])).squeeze(),
        #    self.state
        #)
        
        self.state[pos] = upd[pos].to(dtype=torch.int8).clone()
        #indices = list(zip(np.where(ind_test.numpy()==1)[0], np.where(ind_test.numpy()==1)[1])) 
        #for ind in indices:
        #    prob = torch.Tensor([self.neigh_prob[ind], 1-self.neigh_prob[ind]])
        #    logits = prob.log()
            #logits = F.gumbel_softmax(logits=logit, tau=tau, hard=False)
            #for a in range(1,50):
            #    logits += F.gumbel_softmax(logits=logit, tau=tau, hard=False)
            #logits = logits/50
        #    self.state[ind] = F.gumbel_softmax(logits=logits, hard=True)[0] 


        #self.cont = torch.where(
        #    self.state==1,
        #    self.cont + 1,
        #    self.cont
        #)

        self.cont[self.state==1] += 1
        #id_x = np.where(self.state==1)[0]
        #id_y = np.where(self.state==1)[1]
        #self.ind = list(zip(id_x, id_y))
        self.ind = self.state.eq(1).nonzero().tolist()
        self.susceptibles = torch.sum(self.state==0).item()
        self.infecteds = torch.sum(self.state==1).item()
        self.deads = torch.sum(self.state==2).item()
    
    def write_df(self, step):
        self.df.Susceptibles[step] = self.susceptibles
        self.df.Infecteds[step] = self.infecteds
        self.df.Deads[step] = self.deads
        
    def Expansion(self, seed_value=2022, input=False, data=None, tau=1):
        
        torch.random.manual_seed(seed_value)
        np.random.seed(seed_value)

        if self.N%2 == 0:
            N += 1
        if input == False:
            Theta = torch.from_numpy(np.random.uniform(low=0, high=2*np.pi, size=self.K))
            Rho = torch.from_numpy(np.random.uniform(low=0, high=1, size=self.K))
        else:
            Theta = torch.Tensor(data.Theta.values.astype('float64'))
            Rho = torch.Tensor(data.Rho.values.astype('float64'))
        
        self.S = torch.zeros(self.N, self.N, self.K+1, dtype=torch.int8)
        self.P = torch.zeros(self.N, self.N, self.K, dtype=torch.float64)
        self.S[:, :, 0] = self.state.clone()
        self.__param__(inc=self.inc, partition=self.partition, p0=self.p0, div=self.div)

        # adding columns Theta, Q, Xi, Rho, m
        self.df.Theta[1:] = Theta
        self.q = (Theta/(torch.pi/2) + 1).type(torch.int8)
        self.df.Quadrant[1:] = self.q
        self.xi = torch.where(
            torch.logical_or(self.q == 1, self.q == 3),
            Theta - ((self.q-1)/2)*torch.pi,
            (self.q/2)*torch.pi - Theta
        )
        self.df.Xi[1:] = self.xi
        self.df.Rho[1:] = Rho
        self.m = torch.where(
            Rho <= self.partition[1],
            Rho/self.partition[0],
            Rho/self.partition[2] + 2
        ).type(torch.int8)
        self.df.m[1:] = self.m
        
        self.submatrix()
        self.enlargement_process()

        for L in range(self.K):
            self.neighbourhood_relation(step=L)
            self.P[:, :, L] = self.neigh_prob.clone()
            self.update(tau=tau)
            self.S[:, :, L+1] = self.state.clone()
            self.write_df(step=L+1)

    def MonteCarlo(self, n_it = 10**3, input=False, data=None):

        self.X0 = torch.zeros(self.N, self.N, self.K + 1, dtype=torch.float64)
        self.X1 = torch.zeros(self.N, self.N, self.K + 1, dtype=torch.float64)
        self.X2 = torch.zeros(self.N, self.N, self.K + 1, dtype=torch.float64)
        self.df_MC = pd.DataFrame(torch.zeros(self.K + 1, len(self.columnas), dtype=torch.float64),
                                  columns=self.columnas
                                  )

        if input==False:
            self.__param__()
            self.Expansion(seed_value=0)
            self.dataMC = self.df[['Theta', 'Rho']].drop(0).copy()
                        
        for seed in range(n_it):
            self.__init__(N=self.N, K=self.K)
            self.__param__()
            self.Expansion(seed_value=seed, input=True, data=self.dataMC)
            self.X0 += (self.S==0)*1.
            self.X1 += (self.S==1)*1.
            self.X2 += (self.S==2)*1.
            self.df_MC += self.df.copy()
        
        self.X0 = self.X0/n_it
        self.X1 = self.X1/n_it
        self.X2 = self.X2/n_it
        self.df_MC = self.df_MC/n_it
    
def SpreadModel(seed_value=2022, N=7, K=5, inc=1, partition=[0.1, 0.5, 0.9], p0=0.25, div = 2,
                input=False, data=None, tau=1):

    torch.random.manual_seed(seed_value)
    np.random.seed(seed_value)
    if N%2 == 0:
        N += 1
    if input == False:
        Theta = np.random.uniform(low=0, high=2*np.pi, size=K)
        Rho = np.random.uniform(low=0, high=1, size=K)
    else:
        Theta = list(data.Theta)
        Rho = list(data.Rho)
    E, P = init(N=N, K=K)
    grid = Grid(N=N, K=K)
    E[:, :, 0] = grid.state.clone()
    grid.__param__(inc=inc, partition=partition, p0=p0, div=div)

    for L in range(K):
        theta, rho = Theta[L], Rho[L]
        grid.submatrix(theta=theta)
        grid.enlargement_process(theta=theta, rho=rho)
        grid.neighbourhood_relation()
        grid.update(inc=inc, tau=tau)
        grid.write_df(step=L+1)
        E[:, :, L+1] = grid.state.clone()
        P[:, :, L] = grid.neigh_prob.clone()    
    return E, P, grid.df
