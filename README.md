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

![Jump-diffusion process](/other/O-U_plot.png)

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
![Jump-diffusion process](/other/O-U_drift_diffusion.png)


# A bivariate jump-diffusion process
A bivariate jump-diffusion process, a two-dimensional process with higher-order statistical moments, comprises interaction between both the noise terms from the Wiener process ![Wiener process](/other/W.png) and the Poissonian jumps terms ![Poissonian jump process](/other/J.png) across dimensions.

![Jump-diffusion process](/other/JD_process.png)

An integration scheme as mentioned above allows one to numerically simulate the process.

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
