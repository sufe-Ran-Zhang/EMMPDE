# MMPDE

References:

MMPDE：https://github.com/YangYuSCU/MMPDE-Net

Summary on 2025. 06.12

1. 不管是一维还是二维的, 注意右端项 f(x,t) 的影响,加了右端项方程能量是不守恒的, 但是方程可以有解析解. 大家做实验的时候加不加右端项的都可以做, 不加的话, 没有解析解也没关系, 我们和 PINN 比较就行; 

 ![WechatIMG337](https://github.com/user-attachments/assets/c3c56a83-6bf4-4522-8ebb-79550cab3788)

 ![Screenshot 2025-06-12 at 10 18 28](https://github.com/user-attachments/assets/62babaf6-d91f-41e3-bd18-5f071a2a3d7a)

(王东江继续做这个例子)



2. 二维波动方程在做的话, 可以直接不做加右端项的方程, 这个应该是比较简单的例子. 可以和有限差分以及 PINN 进行对比.

   二维波动方程的程序: https://ww2.mathworks.cn/matlabcentral/fileexchange/62204-2d-wave-equation-simulation
                    https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php

   (徐龙哲做这个二维波动方程的例子)

3. 在多个方法里都表明, 关于时间二阶导数的, 要把 u 关于 t 的一阶导数看出新的函数, 

   ![Screenshot 2025-06-12 at 10 25 42](https://github.com/user-attachments/assets/c9c072a5-ea70-4428-bb6f-0377b5512361)

后续新的程序都按这个方式实现.





