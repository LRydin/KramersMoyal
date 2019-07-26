# KramersMoyal
Python KM is a  python package designed to obtain the Kramers–Moyal coefficients, or conditional moments, from stochastic data of any dimension. It employs kernel density estimations, instead of a histogram approach, to ensure better results for low number of points as well as allowing better fitting of the results

# Changelog
Version 0.31 - Corrections to the fft triming after convolution
Version 0.3 - The major breakthrough: Calculates the Kramers–Moyal coefficients for data of any dimension
Version 0.2 - Introducing convolutions and `gaussian` and `ùniform` kernels. Major speed up in the calculations.
Version 0.1 - One and two dimensional Kramers–Moyal coefficients with an `epanechnikov` kernel


# Installation
A the current stage of the library there is no direct installation protocol. Please download the two `.py` files, `kernels.py` and `km.py` to your working python directory. If you are solely using this library for occasional calculations, you can simply add the files to the directory you are working on, and add them to your import preamble
```python
from kramersmoyal import kernels, km
```

# A one-dimensional stochastic process
## The theory
Take for example the well documented one-dimension Ornstein–Uhlenbeck process, also known as Va&#353;&#237;&#269;ek process, see [here](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). This process is governed by two main parameters: the mean-reverting parameter &theta; and the diffusion parameter &sigma;

![Ornstein–Uhlenbeck process](/other/OU_eq.png)

which can be solved in various ways. For our purposes, recall that the drift coefficients, i.e., the first-order Kramers–Moyal coefficient is given by ![first-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_1.png) and the second-order Kramers–Moyal coefficient is ![second-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_2.png).

Generate an exemplary Ornstein–Uhlenbeck process with your favorite integrator, e.g., the [Euler–Maruyama](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) or with a more powerful tool from [`JiTCSDE`](https://github.com/neurophysik/jitcsde) found on GitHub.
For this example lets take &theta;=.4 and &sigma;=1, over a total time of 100 units, with a sampling of 100 Hertz, and from the generated data series retrieve the two parameters, the drift &theta; and diffusion &sigma;

## Integrating an Ornstein–Uhlenbeck process
Here is a short code on generating a Ornstein–Uhlenbeck stochastic trajectory with a simple Euler–Maruyama integration method

```python
# integration time and time sampling
t_final=50
delta_t=0.0001

# The parameters theta and sigma
theta=.4
sigma=.1

# The time array of the trajectory
time=np.linspace(0,t_final,t_final*int(1/delta_t))

# Initialise the array y
y = np.zeros([time.size])

#Generate a Wiener process
dw = np.random.normal(loc=0, scale=np.sqrt(delta_t),size=[time.size,1])

# Let the process start far from zero
y[0]=2.3

# Integrate the process
for i in range(1,time.size):
    y[i] = y[i-1] - theta*y[i-1]*delta_t + sigma*dw[i]
```

From here we have a plain example of an Ornstein–Uhlenbeck process, starting at ```y[0]=2.3``` and drifting, due to the mean-reverting drift &theta;, down to zero. The effect of the noise effect can be seen across the whole trajectory.

![Jump-diffusion process](/other/O-U_plot.png)

## Using Python KM
Take the timeseries ```y``` and lets study the Kramers–Moyal coefficients after the transient drift back to the zero mean, i.e., for ```time=[20,50]```.



# A bivariate jump-diffusion process
A bivariate jump-diffusion process, a two-dimensional process with higher-order statistical moments, comprises interaction between both the noise terms from the Wiener process ![Wiener process](/other/W.png) and the Poissonian jumps terms ![Poissonian jump process](/other/J.png) across dimensions.

![Jump-diffusion process](/other/JD_process.png)

An integration scheme as mentioned above allows one to numerically simulate the process.

##
