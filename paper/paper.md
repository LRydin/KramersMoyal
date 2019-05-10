---
title: 'Kramers--Moyal coefficients: all orders, any dimension and with kernel-based regression'
tags:
  - Python
  - Julia
  - Stochastic processes
  - Fokker--Planck
authors:
  - name: Leonardo Rydin Gorjão
    orcid: 0000-0001-5513-0580
    affiliation: "1, 2"
  - name: Francisco Meirinhos
    orcid: 0000-0000-0000-0000
    affiliation: 3
affiliations:
 - name: Institute for Theoretical Physics, University of Cologne, 50937~K\"oln, Germany
   index: 1
 - name: Forschungszentrum J\"ulich, Institute for Energy and Climate Research - Systems Analysis and Technology Evaluation (IEK-STE), 52428~J\"ulich, Germany
   index: 2
 - name: Helmholtz Institute for Radiation and Nuclear Physics, University of Bonn, Nussallee~14--16, 53115~Bonn, Germany
   index: 3
date: 18 February 2019
bibliography: bib.bib
---

# Summary

A general problem for measuring and evaluating any stochastic process is the retrieval from data or time-series of the Kramers--Moyal coefficients $\mathcal{M}$, components of the master equation describing the stochastic process.

Given a set of continuous and stationary data, i.e., ergodic or quasi-stationary stochastic data, the extensive literature of stochastic processes awards a set of measures, as the Kramers--Moyal coefficients [CITE RISKEN] or the conditional moments, that link stochastic processes to a probabilistic description of the process or of the family of processes, i.e., to a set of partial differential equations, e.g. the master equation or the Fokker--Planck equations. [CITE CITE]

To unravel the Kramers--Moyal coefficients, the most straightforward approach is to perform a histogram-based regression to evaluate the conditional moments of the system at hand.
Such approach is conventional and requires the most commonly used packages in Python (i.e., `numpy`,  `scipy`, or `pandas`), and is directly embodied in MATLAB, Julia or R, usually under the name of `hist` or `histogram`.

The package presented here comprises a manifold of options: A general open-source toolbox for the calculation of Kramers--Moyal coefficients in any dimension, up to any order, and with a variety of different kernels, with the added freedom to implement a specific kernel, if desired.



# Mathematics

%%% text corrected until here

The probability that an $n$-dimensional state variable $\boldsymbol{x}'(t)\in\mathbb{R}^n$ is observed at position $\boldsymbol{x}$ is given by the conditional probability of the previous (or future) states of the system [Cite RISKEN].
This probabilistic description takes the form

$$\mathcal{M}^{[\ell]}(\boldsymbol{x},t)=\int  dx'(\boldsymbol{x}(t)'-\boldsymbol{x}(t))^\ell W(\boldsymbol{x}'|\boldsymbol{x}),
$$
with $W(\boldsymbol{x}'|\boldsymbol{x})$ the transition probability rate.

The exact evaluation of the Kramers--Moyal coefficients for discrete or discretised datasets $\boldsymbol{y}(t)$ --- any human measure of a process is discrete, as well as any computer generated data --- is bounded by the timewise limit imposed.
Taking as an example a two-dimentional case with $\boldsymbol{x}(t)=(x\_1(t),x\_2(t))\in\mathbb{R}^{2}$, the Kramers--Moyal coefficients $\mathcal{M}^{[\ell, m]}\in\mathbb{R}^{2}$ take the form
\begin{equation}
    \begin{aligned}
        \mathcal{M}^{[\ell, m]}(x\_1,x\_2)=
\lim\_{\Delta t\to 0} \frac{1}{\Delta t} \int & \mathrm{d} y\_{1} (y\_1(t+\Delta t)-y\_1(t))^\ell P(y\_1(t+\Delta t))|\_{y\_1(t)=x\_1} \times \\\ & \mathrm{d} y\_2 (y\_2(t+\Delta t)-y\_2(t))^m P(y\_2(t+\Delta t))|\_{y\_2(t)=x\_2}\nonumber
    \end{aligned}
\end{equation}
at a certain reference measure point $(x\_1,x\_2)$. The Kramers--Moyal coefficents $\mathcal{M}^{\ell, m}(x\_1,x\_2)$ are obtained from a two-dimensional time-series $(y\_1(t),y\_2(t))$.

Theoretically $\Delta t$ should take the limiting case of $\Delta t \to 0$, but the restriction of any measuring or storing device -- or the nature of the observables themselves -- permits only time-sampled or discrete recordings.
The relevance and importance of adequate time-sampling was extensively studied and discussed in [CITE LENA, LEHNERTZ, TABAR].
In the limiting case where $\Delta t$ is equivalent to the sampling rate of the data, the Kramers--Moyal coefficients take the form
$$
\mathcal{M}^{\ell, m}(x\_{1}, x\_{2}) = \frac{1}{\Delta t}  \langle \Delta y\_1^{\ell} \Delta y\_2^{m} |\_{y\_1(t)=x\_1, y\_2(t)=x\_2} \rangle, \mathrm{with}~ \Delta y\_i =  y\_i(t+ \Delta t) - y\_i(t).    
$$


Furthermore, the order of the Kramers--Moyal coefficients is given here by the superscript $\ell$ and $m$.
For such measure of the Kramers--Moyal coefficients, i.e., a probabilistic measure, a probabilistic space exists, assigned to the process, stemming from the master equation describing the family of such processes.
The conventional procedure is to utilise a histogram regression of the observed process and retrieve, via approximation or fitting, the Kramers--Moyal coefficient [CITE original KM measures, and maybe some recent].
The choice of a histogram measure for the Kramers--Moyal coefficient results in an acceptable measure of the probability density functions of the process but requires a new mathematical space (a distribution space).
The usage of a kernel approach, followed here, permits an identical overview without the necessity of a new (descritised) distribution space, given that the equivalent space of the observable can be taken.
That is, there exists a map $\phi$, for $\boldsymbol{x}\in \mathbb{V}$ and $\boldsymbol{y} \in \mathbb{W}$, where both $\mathbb{V}$ and $\mathbb{W}$ are vector spaces, such that $\phi: \mathbb{V} \mapsto \mathbb{W}$ is a homomorphism. In general, $\mathbb{V} \sim \mathbb{W}$, that is, the spaces are isomorphic [IS IT POSSIBLE TO TALK ABOUT AN ISOMORPHISM HERE AS WELL?].

Alike the histogram approach for the measure of the Kramers--Moyal coefficients, each single measure of the observable $\boldsymbol{y}(t)$ is averaged, with a designed weight, into the distribution space.
The standing difference, in comparison to the histogram approach, is the riddance of a (discrete) binning system.
All points are averaged, in a weighted fashion, into the distribution space -- aiding specially in cases where the data sets is scarce -- and awarding a continuous measurable space (easier for fitting, for example).

The choice of the kernel, i.e., the weighting function used for averaging, is a matter of taste.
Equivalently the usage of a fixed-size box shape would closely resemble binning, as in the histogram case.
The Epanechnikov kernel [@Epanechnikov], finite in size, is given by
$$
K(u,v)=\left(\frac{9}{16}\right)\left(1-u^2\right)\left(1-v^2\right), \mathrm{with}~|u|,|v|\leq 1,
$$
is used.
It provides a simple weighted fitting, without the long tails (thus necessarily a wider measure space) of a Gaussian shaped kernel or something similar.

# References
