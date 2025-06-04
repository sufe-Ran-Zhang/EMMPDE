import sys
import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
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
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class sampling_MMPDE():
    def __init__(self, X_f, u, layers, lb, ub, nu, AdamIter, LBFGSIter):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # 修改数据维度为3D
        self.t_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.y_f = torch.tensor(X_f[:, 2:3], requires_grad=True).float().to(device)
        self.fun = u

        self.layers = layers
        self.nu = nu

        self.dnn = DNN(layers).to(device)

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(),
                                           lr=1e-3,
                                           betas=(0.9, 0.999),
                                           eps=1e-8)
        
        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.5,
            max_iter=LBFGSIter)

        self.AdamIter = AdamIter
        self.iter = 0
        self.start_time = None
    def detach(self, data):
        return data.detach().cpu().numpy()
        
    def monitor(self, u, ux, uy):
        w = (1 + (ux)**2 + (uy)**2)**(1/2)
        return w

    def net_sample(self, t, x, y):
        xyNew = self.dnn(torch.cat([t, x, y], dim=1))
        xNew = xyNew[:, 0:1]
        yNew = xyNew[:, 1:2]
        
        g0x = x - self.lb[1]
        g1x = x - self.ub[1]
        g0y = y - self.lb[2]
        g1y = y - self.ub[2]
        
        xNew = g0x*g1x*xNew + x
        yNew = g0y*g1y*yNew + y
        
        return xNew, yNew

    def net_f(self, t, x, y):
        xNew, yNew = self.net_sample(t, x, y)

        # 计算时间导数
        xNew_t = torch.autograd.grad(
            xNew, t,
            grad_outputs=torch.ones_like(xNew),
            retain_graph=True,
            create_graph=True
        )[0]
        
        yNew_t = torch.autograd.grad(
            yNew, t,
            grad_outputs=torch.ones_like(yNew),
            retain_graph=True,
            create_graph=True
        )[0]

        # 计算空间导数
        xNew_x = torch.autograd.grad(
            xNew, x,
            grad_outputs=torch.ones_like(xNew),
            retain_graph=True,
            create_graph=True
        )[0]
        
        xNew_y = torch.autograd.grad(
            xNew, y,
            grad_outputs=torch.ones_like(xNew),
            retain_graph=True,
            create_graph=True
        )[0]
        
        yNew_x = torch.autograd.grad(
            yNew, x,
            grad_outputs=torch.ones_like(yNew),
            retain_graph=True,
            create_graph=True
        )[0]
        
        yNew_y = torch.autograd.grad(
            yNew, y,
            grad_outputs=torch.ones_like(yNew),
            retain_graph=True,
            create_graph=True
        )[0]

        # 计算二阶导数
        xNew_xx = torch.autograd.grad(
            xNew_x, x,
            grad_outputs=torch.ones_like(xNew_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        xNew_yy = torch.autograd.grad(
            xNew_y, y,
            grad_outputs=torch.ones_like(xNew_y),
            retain_graph=True,
            create_graph=True
        )[0]
        
        yNew_xx = torch.autograd.grad(
            yNew_x, x,
            grad_outputs=torch.ones_like(yNew_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        yNew_yy = torch.autograd.grad(
            yNew_y, y,
            grad_outputs=torch.ones_like(yNew_y),
            retain_graph=True,
            create_graph=True
        )[0]

        u = self.fun(torch.cat([t, x, y], dim=1))

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True)[0]
            
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            create_graph=True)[0]

        G = self.monitor(u, u_x, u_y)

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
        
        G_y = torch.autograd.grad(
            G, y,
            grad_outputs=torch.ones_like(G),
            retain_graph=True,
            create_graph=True
        )[0]

        # MMPDE方程 
        # x方向上的网格变形项
        Ex = G_x * xNew_x + G_y * xNew_y + G * (xNew_xx + xNew_yy) #监控函数梯度与网格点位置一阶导数的乘积 监控函数与网格点位置二阶导数的乘积
        Ey = G_x * yNew_x + G_y * yNew_y + G * (yNew_xx + yNew_yy) 
        
        fx = xNew_t*self.nu*(G**2)*(xNew_x**2 + xNew_y**2) + Ex
        fy = yNew_t*self.nu*(G**2)*(yNew_x**2 + yNew_y**2) + Ey

        return fx, fy


    def loss_func(self):
        fx_pred, fy_pred = self.net_f(self.t_f, self.x_f, self.y_f)
        loss_fx = torch.mean(fx_pred ** 2)
        loss_fy = torch.mean(fy_pred ** 2)
        
        loss = loss_fx + loss_fy

        return loss, loss_fx + loss_fy
    
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
        self.train_Adam(self.optimizer_Adam, self.AdamIter, None)
        print("MMPDE_Adam done!")
        self.train_LBFGS(self.optimizer_LBFGS, None)
        print('MMPDE_LBGFS done!')

        self.dnn.eval()
        print("1234")
        new_x, new_y = self.net_sample(self.t_f, self.x_f, self.y_f)
        # 添加维度检查
        print(f"t_f shape: {self.t_f.shape}")
        print(f"new_x shape: {new_x.shape}")
        print(f"new_y shape: {new_y.shape}")
        new_sample = torch.cat([self.t_f, new_x, new_y], dim=1)
        print(f"new_sample shape: {new_sample.shape}")
        return new_sample

    def predict(self, X):
        t = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.dnn.eval()
        x_new, y_new = self.net_sample(t, x, y)
        fx, fy = self.net_f(t, x, y)
        
        x_new = x_new.detach().cpu().numpy()
        y_new = y_new.detach().cpu().numpy()
        fx = fx.detach().cpu().numpy()
        fy = fy.detach().cpu().numpy()
        
        return x_new, y_new, fx, fy

def function_Tanh(x):
    p = 50
    tanh = torch.nn.Tanh()
    # 修改为3D测试函数
    u = tanh(p * ((x[:, 1] - x[:, 0])**2 + (x[:, 2] - x[:, 0])**2))
    return u

if __name__ == "__main__":
    nu = 1
    adam_iter_MM, lbgfs_iter_MM = 10, 10
    
    # 修改网络结构：输入3维，输出2维
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    # 生成3D网格数据
    t = np.linspace(0, 1, 21)
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    
    T, X, Y = np.meshgrid(t, x, y)
    X_star = np.hstack((T.flatten()[:, None], 
                       X.flatten()[:, None],
                       Y.flatten()[:, None]))

    # 边界范围
    lb = X_star.min(0)  # [t_min, x_min, y_min]
    ub = X_star.max(0)  # [t_max, x_max, y_max]

    X_f_train = X_star

    model = sampling_MMPDE(X_f_train, function_Tanh, layers, lb, ub, nu, 
                          adam_iter_MM, lbgfs_iter_MM)
    new_sample = model.train()

    # 预测和可视化
    x_new, y_new, fx, fy = model.predict(X_star)
    
    # 保存结果
    scipy.io.savemat('3D_results.mat', 
                    {'x_new': x_new, 'y_new': y_new, 'fx': fx, 'fy': fy})

    # 可视化示例（选择特定时间片）
    t_idx = 0  # 选择第一个时间片
    n_points = 21  # 每个维度的点数
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_new[t_idx*n_points**2:(t_idx+1)*n_points**2], 
               y_new[t_idx*n_points**2:(t_idx+1)*n_points**2], 
               c='b', s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f't = {t[t_idx]}')
    plt.show()