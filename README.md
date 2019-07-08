# KramersMoyal
KramersMoyal is a python package designed to recover the Kramers–Moyal coefficients from data. The package is currently capable of recovering one- and two-dimensional Kramers–Moyal coefficients, i.e., from data with one or two dimensions

# Installation
A the current stage of the library there is no direct installation protocol. Please download the two `.py` files, `kernels.py` and `km.py` to your working python directory. If you are solely using this library for occasional calculations, you can simply add the files to the directory you are working on, and add them to your import preamble
```python
from kramersmoyal import kernels, km
```

# A one-dimensional stochastic process
## The theory
Take for example the well documented one-dimension Ornstein–Uhlenbeck process, see [here](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). This process is governed by two main parameters: the mean-reverting parameter &theta; and the diffusion parameter &sigma;

![Ornstein–Uhlenbeck process](/other/OU_eq.png)

which can be solved in various ways. For our purposes, recall that the drift coefficients, i.e., the first-order Kramers–Moyal coefficient is given by ![first-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_1.png) and the second-order Kramers–Moyal coefficient is ![second-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_2.png).

Generate an exemplary Ornstein–Uhlenbeck process with your favorite integrator, e.g., the [Euler–Maruyama](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) or with a more powerful tool from [`JiTCSDE`](https://github.com/neurophysik/jitcsde) found on GitHub.
For this example lets take &theta;=2 and &sigma;=1, over a total time of 100 units, with a sampling of 100 Hertz, and from the generated data series retrieve the two parameters, the drift &theta; and diffusion &sigma;

## Using the library
Here is a short code on generating stochastic trajectories with a simple Euler–Maruyama integration method

```python
# integration time and time sampling
time=25
delta_t=0.001

# The parameters theta and sigma
theta=.5  
sigma=.25

# The time array of the trajectory
time=np.linspace(0,t_final,t_final*int(1/delta_t))

# Initialise the array y
y = np.zeros([time.size])

#Generate a Wiener process
dw = np.random.normal(loc=0, scale=np.sqrt(delta_t),size=[time.size,1])

# Give some random initial conditions far from zero
y[0]=np.random.normal(size=1)/100 + 2.3

# Integrate the process
for i in range(1,time.size):
    y[i] = y[i-1] - theta*y[i-1]*delta_t + sigma*dw[i]
```


# Bivariate jump-diffusion process
To illustrate now a more complicated process, we present here a bivariate jump-diffusion process, a two-dimensional process with higher-order statistical moments, that allows for interaction between both the noise terms from the Wiener process ![Wiener process](/other/W.png) and the Poissonian jumps terms ![Poissonian jump process](/other/J.png) across dimensions.

![Jump-diffusion process](/other/JD_process.png)

Although there are several parameters in the system, it is still possible to partially recover them. For the presented case, we want to evaluate the recovery of the Kramers–Moyal coefficients strictly from data, without any a priori assumptions on the data.
