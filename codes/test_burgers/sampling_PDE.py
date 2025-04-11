import sys

sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

#from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        #x_min = x.min(dim=0, keepdim=True)[0]
        #x_max = x.max(dim=0, keepdim=True)[0]
        #x = 2.0 * ((x - x_min) / (x_max - x_min)) - 1.0
        #print('x:',x)
        out = self.layers(x)
        return out


# the physics-guided neural network
class sampling_MMPDE():
    def __init__(self, X_f, u, layers, lb, ub, nu, AdamIter, LBFGSIter):
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        #self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
        #self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        #self.u = torch.tensor(u, requires_grad=True).float().to(device)
        self.fun = u

        #self.gard_x = grad_x

        self.layers = layers
        self.nu = nu

        # deep neural networks
        self.dnn = DNN(layers).to(device)

        # optimizers: using the same settings
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(),
                                               lr=1e-3,
                                               betas=(0.9, 0.999),
                                               eps=1e-8,
                                               weight_decay=0,
                                               amsgrad=False)

        self.AdamIter = AdamIter

        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.5,
            max_iter=LBFGSIter,
            #max_eval=100,
            #history_size=100,
            #tolerance_grad=-1,
            #tolerance_change=1.0 * np.finfo(float).eps,
            #line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )

        self.optimizer = None
        self.loss = None
        self.iter = 0
        self.start_time = None

    def detach(self, data):
        return data.detach().cpu().numpy()


    def fun_x(self, u, x):
        dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] # du/dx  size([,1])
        #print("size of dx:", dx)
        return dx

    def monitor(self, u, ux):
        #w = (1 + u ** 2 + (0.5*ux) ** 2) ** (1 / 2)
        w = (1 + (ux) ** 2) ** (1 / 2)
        return w

    def net_sample(self, t, x):
        xNew = self.dnn(torch.cat([t, x], dim=1))
        #g0 = (1 - torch.exp(-(x - self.lb[1])))
        #g1 = (1 - torch.exp(-(x - self.ub[1])))
        #g2 = (1 - torch.exp(-t))
        g0 = x - self.lb[1]
        g1 = x - self.ub[1]
        xNew = g0*g1*xNew + x
        #print('new x:', xNew)
        return xNew

    def net_f(self, t, x):

        xNew = self.net_sample(t, x) #size([,1])
        #print('size xnew:', xNew.shape)

        xNew_t = torch.autograd.grad(
            xNew, t,
            grad_outputs=torch.ones_like(xNew),
            retain_graph=True,
            create_graph=True
        )[0]

        xNew_x = torch.autograd.grad(
            xNew, x,
            grad_outputs=torch.ones_like(xNew),
            retain_graph=True,
            create_graph=True
        )[0]
        xNew_xx = torch.autograd.grad(
            xNew_x, x,
            grad_outputs=torch.ones_like(xNew_x),
            retain_graph=True,
            create_graph=True
        )[0]


        '''
        u = self.fun(xNew, t)
        u_x = self.fun_x(xNew, t)
        u_xx = self.fun_xx(xNew, t)
        u_xt = self.fun_xt(xNew, t)
        '''
        u = self.fun(torch.cat([t, x], dim=1)) #size([,1])
        #print('u:', u)
        #print('size u:', u.shape)
        #print('u in mmpde:', u)


        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True)[0]  # du/dx  size([,1])

        #u_x = self.fun_x(u, x)

        G = self.monitor(u, u_x)


        G_t = torch.autograd.grad(
            G, t,
            grad_outputs=torch.ones_like(G),
            retain_graph=True,
            create_graph=True
        )[0]

        G_x = torch.autograd.grad(
            G, x,
            grad_outputs=torch.ones_like(G),
            retain_graph=True,
            create_graph=True
        )[0]

        #print('size G_x:', G_x.shape)


        G_tx = torch.autograd.grad(
            G_t, x,
            grad_outputs=torch.ones_like(G_t),
            retain_graph=True,
            create_graph=True
        )[0]

        E = G_x * xNew_x + G * xNew_xx
        f = xNew_t*self.nu*(G**2)*(xNew_x**2) + E    # succuss, nu =1

        E_t = G_tx * xNew_x + G_t * xNew_xx
        #f = xNew_t*self.nu*(G**2)*(xNew_x**2) + E + E_t     # also work, but with larger residuals, nu=1


        #f_np = self.detach(f)
        #print('loss of equ:',f)
        #scipy.io.savemat('loss_equ.mat', {'loss_equ': f_np})

        return f

    def loss_func(self):

        #u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.t_f, self.x_f)
        #loss_u = torch.mean((self.u - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        #loss = loss_u + loss_f
        alpha = 1
        loss = alpha*loss_f

        return  loss, loss_f
        #return loss, loss_u, loss_f

    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # Loss function initialization
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(device)
        self.loss.requires_grad_()

        #self.loss, self.loss_u, self.loss_f = self.loss_func()
        self.loss, self.loss_f = self.loss_func()
        self.loss.backward()
        self.iter = self.iter + 1

        if self.iter % 100 == 0:
            loss = self.detach(self.loss)
            #loss_u = self.detach(self.loss_u)
            loss_equ = self.detach(self.loss_f)

            #log_str = str(self.optimizer_name) + ' Iter ' + str(self.iter) + ' Loss ' + str(loss) + \
            #          ' loss of function ' + str(loss_u) + ' loss of equ ' + str(loss_equ)
            log_str = str(self.optimizer_name) + ' Iter ' + str(self.iter) + ' Loss ' + str(loss) + \
                      ' loss of equ ' + str(loss_equ)
            print(log_str)

            elapsed = time.time() - self.start_time
            print('MMPDE_Iter 10, Time: %.4f' % (elapsed))
            torch.save(self.dnn.state_dict(), 'MMPDE.pth')

            self.start_time = time.time()

        return self.loss

    def train_Adam(self, optimizer, nIter, Adam_scheduler):
        self.dnn.train()

        self.optimizer = optimizer
        self.optimizer_name = 'MMPDE_Adam'
        self.scheduler = Adam_scheduler
        for it in range(nIter):
            #print('iter_adam:', it)
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(self.loss)

    def train_LBFGS(self, optimizer, LBFGS_scheduler):
        self.dnn.train()

        self.optimizer = optimizer
        self.optimizer_name = 'MMPDE_LBFGS'
        self.scheduler = LBFGS_scheduler

        def closure():
            loss = self.optimize_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss

        self.optimizer.step(closure)

    def train(self):

        # Backward and optimize
        self.train_Adam(self.optimizer_Adam, self.AdamIter, None)
        print("MMPDE_Adam done!")
        self.train_LBFGS(self.optimizer_LBFGS, None)
        print('MMPDE_LBGFS done!')
        #torch.save(self.dnn.state_dict(), 'model/Tanh.pth')


        ## return: the new samples
        self.dnn.eval()
        new_x = self.net_sample(self.t_f, self.x_f)  # new location
        new_sample = torch.cat([self.t_f, new_x], dim=1)
        return new_sample

    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_sample(t, x)
        f = self.net_f(t, x)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f



def function_Tanh(x):
    p = 50
    tanh = torch.nn.Tanh()
    u = tanh(p * (x[:, 1] - x[:, 0]))   #u=tanh(p*(x-t))
    return u


def function_exp(x):
    u = torch.exp(-(x[:, 1] - x[:, 0]) ** 2 / 0.01)
    return u

if __name__ == "__main__":
    nu = 1

    adam_iter_MM, lbgfs_iter_MM = 10, 10


    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('data/Tanh.mat')
    #data = scipy.io.loadmat('Tanh.mat')  #GPU

    x = data['x'].flatten()[:, None]

    t = data['t'].flatten()[:, None]
    #Exact = np.real(data['usol'])

    T, X = np.meshgrid(t, x)
    X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))

    x_inside = x
    T_inside, X_inside = np.meshgrid(t, x_inside)

    TX_inside = np.hstack((T_inside.flatten()[:, None], X_inside.flatten()[:, None]))
    #u_star = Exact.flatten()[:,None]

    #print('X_star=', X_star)

    # Doman bounds
    lb = X_star.min(0) #[0,0] #[t,x]
    ub = X_star.max(0) #[1,1]

    X_f_train = TX_inside  # all the points including boundaries and initials


    #print('train points of ini:', len(X_u_train))
    print('train points of equ:', X_f_train)


    model = sampling_MMPDE(X_f_train, function_Tanh, layers, lb, ub, nu, adam_iter_MM, lbgfs_iter_MM)
    model.train()

    ## model predict
    data_pred = scipy.io.loadmat('1D_X_Pred.mat')  #GPU
    x_pred = data_pred['X_pred']
    T_pred, X_pred = np.meshgrid(t, x_pred)
    TX_pred = np.hstack((T_pred.flatten()[:, None], X_pred.flatten()[:, None]))
    xi_pred, f_pred = model.predict(TX_pred)
    scipy.io.savemat('xi_pred.mat', {'xi_pred': xi_pred})

    ## ploting
    nt = len(t)
    nx = len(x_pred)

    xi_pred1 = xi_pred.reshape((nt, nx), order='F')
    print(xi_pred1)



    # points distribution at initial time
    xi_t0 = xi_pred1[0, :].flatten()[:, None]
    plt.plot(xi_t0, 0 * xi_t0, 'bo', markersize=1)
    plt.xlabel('$x$', fontsize=20)
    plt.title(r'points distribution at $t_0$', fontsize=20)
    plt.show()



    ## plot the  sampling of the terminal time
    xi_t0 = xi_pred1[-1, :].flatten()[:, None]
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.5, 0.5)
    plt.plot(xi_t0, 0 * xi_t0, 'bo', markersize=1)
    plt.xlabel('$x$', fontsize=20)
    plt.title(r'points distribution at $t_T$', fontsize=20)
    plt.show()


    ## plot the trajectaries of the left and right points
    xi_0 = xi_pred1[:, 0].flatten()[:, None]
    xi_1 = xi_pred1[:, -1].flatten()[:, None]
    plt.plot(xi_0, t, markersize=1)
    plt.plot(xi_1, t, markersize=1)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$t$', fontsize=20)
    plt.title(r'points trajectaries of the left and right points', fontsize=20)
    plt.show()