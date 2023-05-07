from sklearn.base import BaseEstimator
import ot
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cpu'
def project_onto_simplex(p):
    """Project a point onto the simplex"""
    sh = p.shape
    p_flatten = torch.flatten(p)
    x = p_flatten
    n = len(x)
    x_sorted = torch.sort(x,descending=True).values
    cum_sum = torch.cumsum(x_sorted,0)
    t = (cum_sum - 1) / torch.arange(1, n + 1).to(device=device)
    rho = torch.max(torch.where(t >= x_sorted, torch.tensor([0.]).to(device=device),t)).to(device=device)
    w = torch.maximum(x - rho, torch.zeros(n).to(device=device)).to(device=device)

    w = w.reshape(sh)
    return w

def mean1(z):
    return z
def mean2(z,c):
    return z+10*c
def phi(x):
    return x


def grad_desc(x,y,C,hp1,hp2,lr,itr):
	
	# print(x.shape,C.shape)
	m_tot = x.shape[0]
	p = torch.rand((m_tot,m_tot,m_tot)).to(device=device)
	p.data = project_onto_simplex(p)
	losses = []
	p.requires_grad_()
	optimizer = optim.Adam([p],lr=lr)
	# print("loss_bef = ", obj(p,C))

	for i in range(itr):
		f, sub = loss_function(x,y,p,C,hp1,hp2)
		losses.append(f.data.to(device="cpu").numpy())
        # losses.append(f.data.to(device="cpu").numpy())
		f.backward()
		optimizer.step()
		optimizer.param_groups[0]["params"][0].data = project_onto_simplex(optimizer.param_groups[0]["params"][0])
		optimizer.zero_grad()
  
	ans = optimizer.param_groups[0]["params"][0].data
	return ans,losses
   

def loss_function(x,y,P,C,hp1,hp2):
        Pxy = torch.sum(P,dim=2)
        Pxz = torch.sum(P,dim=1)
        Pyz = torch.sum(P,dim=0)

        Pz = torch.sum(Pxz,dim=0)

        f1 = torch.sum(Pxy*C)
        
        f2 = 0
        m_tot = x.shape[0]
        
        for i in range(m_tot):
            if(not Pz[i]):
                continue
            
            # s = 0
            # for j in range(m_tot):
            #     s += (Pxz[j][i]/Pz[i])*phi(x[j])
            # s -= phi(x[i])
            a = 0
            a = torch.matmul(Pxz[:,i]/Pz[i],phi(x))
            a -= phi(x[i])
            f2 += torch.norm(a)**2

        f3 = 0
        for i in range(m_tot):
            if(not Pz[i]):
                continue
            # s = 0
            # for j in range(m_tot):
            #     s += (Pyz[j][i]/Pz[i])*phi(y[j])
            # s -= phi(y[i])
            a = 0
            a = torch.matmul(Pyz[:,i]/Pz[i],phi(y))
            a -= phi(y[i])
            f3 += torch.norm(a)**2

        f = f1 + hp1*f2 + hp2*f3
        return f, (f1,f2,f3)


class MyLossEstimator(BaseEstimator):
    def __init__(self, hp1=0.01, hp2=32):
        self.hp1 = hp1
        self.hp2 = hp2
        
        
    def fit(self, x,y):
        M = ot.dist(x,y)
        C = M.to(device=device)
        P = grad_desc(x,y,C,self.hp1,self.hp2,0.01,100)[0]
        (f1,f2,f3) = loss_function(x,y,P,C,self.hp1,self.hp2)[1]
        self.loss_ = f2 + f3 
        return self
    
    def score(self,X,y):
        return -self.loss_  # Scikit-learn minimizes scores, so we use negative loss


# Define the hyperparameters to search over
param_grid = {'hp1': [1,10,100,500,1000],
              'hp2': [1,10,100,500,1000]}

m = 10
nz = 10
c=0

c_arr = []
f_arr = []
f1_arr = []
f2_arr = []
f3_arr = []

for  c in range(0,100,10):
    mz1 = [10,13]
    xvz1 = np.array([[10,12],[12,13]])
    sz1 = xvz1@xvz1.T
    
    mz2 = [1,3]
    xvz2 = np.array([[1,2],[2,3]])
    sz2 = xvz2@xvz2.T

    z1 = np.random.multivariate_normal(mz1,sz1,nz)
    z2 = np.random.multivariate_normal(mz2,sz2,nz)
    

    xv = np.array([[18,13],[20,15]])
    s = xv@xv.T

    x = np.zeros((1,2))
    y = np.zeros((1,2))


    for i in range(0,nz):
        x = np.vstack((x,np.random.multivariate_normal(mean1(z1[i]), s, m)))
        y = np.vstack((y,np.random.multivariate_normal(mean2(z2[i],c), s, m) ))

    x = x[1:]
    y = y[1:]
    # print(x.shape,y.shape)
    a = [1/(m*nz)]*m*nz
    b = [1/(m*nz)]*m*nz

    m_tot = m*nz
    x = torch.from_numpy(x).to(device=device).float()
    y = torch.from_numpy(y).to(device=device).float()

    estimator = MyLossEstimator()
    grid_search = GridSearchCV(estimator, param_grid)

    grid_search.fit(x,y)

    best_params = grid_search.best_params_
    print(best_params)

    M = ot.dist(x,y)
    C = M.to(device=device)
    P = grad_desc(x,y,C,best_params['hp1'], best_params['hp2'],0.01,100)[0]
    f, (f1,f2,f3) = loss_function(x,y,P,C,best_params['hp1'],best_params['hp1'])
    # print(f1,f2,f3)
    
    c_arr.append(c)
    f_arr.append(f.data.to(device='cpu').numpy())
    f1_arr.append(f1.data.to(device='cpu').numpy())
    f2_arr.append(f2.data.to(device='cpu').numpy())
    f3_arr.append(f3.data.to(device='cpu').numpy())
    
    
plt.figure(1)
plt.plot(c_arr,f_arr)
plt.savefig('Images/diffZMeanVary/f.png')

plt.figure(2)
plt.plot(c_arr,f1_arr)
plt.savefig('Images/diffZMeanVary/f1.png')


plt.figure(3)
plt.plot(c_arr,f2_arr)
plt.savefig('Images/diffZMeanVary/f2.png')


plt.figure(4)
plt.plot(c_arr,f3_arr)
plt.savefig('Images/diffZMeanVary/f3.png')
