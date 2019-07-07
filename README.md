# KramersMoyal
KramersMoyal is a python package designed to recover the Kramers–Moyal coefficients from data. The package is currently capable of recovering one- and two-dimensional Kramers–Moyal coefficients, i.e., from data with one or two dimensions

# A one-dimensional stochastic process
Take for example the well documented one-dimension Ornstein–Uhlenbeck process, see [here](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). This process is governed by two main parameters: the mean-reverting parameter &theta; and the diffusion parameter &sigma;

![Ornstein–Uhlenbeck process](/other/OU_eq.png)

which can be solved in various ways. For our purposes, recall that the drift coefficients, i.e., the first-order Kramers–Moyal coefficient is given by ![first-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_1.png) and the second-order Kramers–Moyal coefficient is ![second-order Kramers–Moyal coefficient of an Ornstein–Uhlenbeck process](/other/KM_2.png).

Generate an exemplary Ornstein–Uhlenbeck process with your favorite integrator, e.g., the [Euler–Maruyama](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) or with a more powerful tool from [`JiTCSDE`](https://github.com/neurophysik/jitcsde) found on GitHub.
For this example lets take &theta;=2 and &sigma;=1, over a total time of 100 units, with a sampling of 100 Hertz, and from the generated data series retrieve the two parameters, the drift &theta; and diffusion &sigma;

# Bivariate jump-diffusion process
To illustrate now a more complicated process, we present here a bivariate jump-diffusion process, a two-dimensional process with higher-order statistical moments, that allows for interaction between both the noise terms from the Wiener process ![Wiener process](/other/W.png) and the Poissonian jumps terms ![Poissonian jump process](/other/J.png) across dimensions.

![Ornstein–Uhlenbeck process](/other/JD_process.png)

Although there are several parameters in the system, it is still possible to partially recover them. For the presented case, we want to evaluate the recovery of the Kramers–Moyal coefficients strictly from data, without any a priori assumptions on the data.
