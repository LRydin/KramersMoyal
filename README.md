# KramersMoyal
Python KM is a  python package designed to obtain the Kramers–Moyal coefficients, or conditional moments, from stochastic data of any dimension. It employs kernel density estimations, instead of a histogram approach, to ensure better results for low number of points as well as allowing better fitting of the results

# Changelog
- Version 0.31 - Corrections to the fft triming after convolution
- Version 0.3 - The major breakthrough: Calculates the Kramers–Moyal coefficients for data of any dimension
- Version 0.2 - Introducing convolutions and `gaussian` and `uniform` kernels. Major speed up in the calculations.
- Version 0.1 - One and two dimensional Kramers–Moyal coefficients with an `epanechnikov` kernel


# Installation
A the current stage of the library there is no direct installation protocol. Just get the `kramersmoyal` into your working python directory and add your import preamble
```python
from kramersmoyal import km, kernels
```

# A one-dimensional stochastic process

A Jupyter notebook with this example can be found [here](/examples/kmc.ipynb)

 ## The theory
Take for example the well documented one-dimension Ornstein–Uhlenbeck process, also known as Va&#353;&#237;&#269;ek process, see [here](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). This process is governed by two main parameters: the mean-reverting parameter &theta; and the diffusion parameter &sigma;

![Ornstein–Uhlenbeck process](/other/OU_eq.png)

which can be solved in various ways. For our purposes, recall that the drift coefficient, i.e., the first-order Kramers–Moyal coefficient, is given by ![first-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_1.png) and the second-order Kramers–Moyal coefficient is ![second-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_2.png), i.e., the diffusion.

Generate an exemplary Ornstein–Uhlenbeck process with your favorite integrator, e.g., the [Euler–Maruyama](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) or with a more powerful tool from [`JiTCSDE`](https://github.com/neurophysik/jitcsde) found on GitHub.
For this example lets take &theta;=.3 and &sigma;=.1, over a total time of 500 units, with a sampling of 1000 Hertz, and from the generated data series retrieve the two parameters, the drift &theta; and diffusion &sigma;.

## Integrating an Ornstein–Uhlenbeck process
Here is a short code on generating a Ornstein–Uhlenbeck stochastic trajectory with a simple Euler–Maruyama integration method

```python
# integration time and time sampling
t_final = 500
delta_t = 0.001

# The parameters theta and sigma
theta = .3
sigma = .1

# The time array of the trajectory
time = np.arange(0,t_final,delta_t)

# Initialise the array y
y = np.zeros(time.size)

#Generate a Wiener process
dw = np.random.normal(loc=0, scale=np.sqrt(delta_t),size=time.size)

# Integrate the process
for i in range(1,time.size):
    y[i] = y[i-1] - theta*y[i-1]*delta_t + sigma*dw[i]
```

From here we have a plain example of an Ornstein–Uhlenbeck process, always drifting back to zero, due to the mean-reverting drift &theta;. The effect of the noise effect can be seen across the whole trajectory.

<img src="/other/fig1.png" title="Ornstein–Uhlenbeck process" height="200"/>

## Using Python KM
Take the timeseries `y` and lets study the Kramers–Moyal coefficients. For this lets look at the drift and diffusion coefficients of the process, i.e., the first and second Kramers–Moyal coefficients, with an `epanechnikov` kernel
```python
# Choose number of points of you target space
bins = np.array([5000])

# Choose powers to calculate
powers = np.array([[1],[2]])

# Choose your desired bandwith
bw = .15

# The kmc hold the results, where edges holds the binning space
kmc, edges = km(y, kernel=kernels.epanechnikov, bw=bw, bins=bins, powers=powers)
```

This results in

<img src="/other/fig2.png" title="Drift and diffusion terms" height="200"/>


# A 2-dimensional diffusion process

A Jupyter notebook with this example can be found [here](/examples/kmc.ipynb)

A 2-dimensional diffusion process is an process that comprises two ![Wiener process](/other/inline_W.png) and allows for a mixing of these noise terms across its two dimensions.

## Theory
A two-dimensionan diffusion process takes the form

<img src="/other/2D-diffusion.png" alt="2D-diffusion" title="A 2-dimensional diffusion process" height="60" />

where we will select a set of more complicated parameters obeying

<img src="/other/parameters_2D-diffusion.png" alt="2D-diffusion" title="A 2-dimensional diffusion process" height="70" />

with ![](/other/inline_parameters_2D-diffusion.png).

## Choice of parameters
As an example, lets take the following set of parameters for the drift vector and diffusion matrix

```python
# Define the drift vector N
N = np.array([2.0, 1.0])

# Define the diffusion matrix g
g = np.array([[0.5, 0.0],[0.0, 0.5]])

# The total integration time
t_start = 0
t_final = 2000

# The desired timestep of integration
delta_t = 0.001

# time array of the process
time = np.linspace(0, t_final, t_final * int(1 / delta_t))
```
## Integrating a 2-dimensional process
Integrating the previous stochastic trajectory with a simple Euler–Maruyama integration method

```python
# Initialise the array y
y = np.zeros([time.size, 2])

# Generate two Wiener processes with a scale of np.sqrt(delta_t)
dW = np.random.normal(loc = 0, scale = np.sqrt(delta_t), size = [time.size, 2])

# Give some small random initial conditions
y[0,:]=np.random.normal(size = 2) / 10

# Integrate the process (takes about 20 secs)
for i in range(1, time.size):
    y[i,0] = y[i-1,0]  -  N[0] * y[i-1,0] * delta_t
             + g[0,0]/(1 + np.exp(y[i-1,0]**2)) * dW[i,0]  +  g[0,1] * dW[i,1]
    y[i,1] = y[i-1,1]  -  N[1] * y[i-1,1] * delta_t
             + g[1,0] * dW[i,0]  +  g[1,1]/(1 + np.exp(y[i-1,1]**2)) * dW[i,1]
```

The stochastic trajectory in 2 dimensions for 10 time units

<img src="/other/fig3.png" alt="2D-diffusion" title="2-dimensional trajectory" height="280" />

## Back to Python KM and the Kramers–Moyal coefficients
First notice that all the results now will be two-dimensional surfaces, so we
will need to plot them as such

```python
# Choose the size of your target space in two dimensions
bins = np.array([300, 300])

# Introduce the desired orders to calculate, but in 2 dimensions
powers = np.array([[0,0], [1,0], [0,1], [1,1], [2,0], [0,2], [2,2]])
# insert into kmc:   0      1      2      3      4      5      6

# Notice that the first entry in [,] is for the first dimension, the
# second for the second dimension...

# Choose a desired bandwidth bw
bw = 0.1

# Calculate the Kramers−Moyal coefficients
kmc, edges = km(y, bw = bw, bins = bins, powers = powers)

# The K−M coefficients are stacked along the first dim of the
# kmc array, so kmc[1,...] is the first K−M coefficient, kmc[2,...]
# is the second. These will be 2-dimensional matrices
```

Now one can visualise the Kramers–Moyal coefficients (surfaces)

<img src="/other/fig4.png" alt="2D-diffusion" title="2-dimensional trajectory" height="480" />

# Contributions
We welcome reviews and ideas for everyone. If you want to share your ideas or report a bug, open a [issue](https://github.com/LRydin/KramersMoyal/issues) here on GitHub, or contact us directly.

# TODOs
Next on the list is
- Include an optimal bandwith calculator, based on [Silverman's](https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator)
- Include more kernels
- Add install script
- Work throught the documentation carefully
- Extend examples here and in the documentation
- Create a sub-routine to calculate the KMs without a convolution

# Support
### History
This project was started in 2017 at the [neurophysik](https://www.researchgate.net/lab/Klaus-Lehnertz-Lab-2) with Klaus Lehnertz, M. Reza Rahimi Tabar, and Jan Heysel. Francisco Meirinhos devised later the hard coding to python. The project is now supported by Dirk Witthaut and the [Institute of Energy and Climate Research Systems Analysis and Technology Evaluation](https://www.fz-juelich.de/iek/iek-ste/EN/Home/home_node.html).

### Funding
Helmholtz Association Initiative _Energy System 2050 - A Contribution of the Research Field Energy_ and the grant No. VH-NG-1025.
