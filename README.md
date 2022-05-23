# Contextual Kernel Elimination

To install the needed libraries, run:

```shell
pip install -r requirements.txt
```

[toc]

## Agents



## Kernels

We implement Gaussian and Exponential kernels. Let's denote $s_1$ and $s_2$ two context-action pairs.

#### Gaussian kernel

$$
\phi(s_1, s_2)= \exp(-\gamma \Vert s_1-s_2\Vert ^2)
$$

with $\gamma = \frac{1}{2(0.1)^2}$

#### Exponential kernel

$$
\phi(s_1, s_2) = \exp(-\gamma \Vert s_1-s_2\Vert)
$$

with $\gamma = 10$

## Environments

We present here the characteristics of the testing environments.

#### Bump Enviromnent

The rewards are generated as:
$$
r(x, a) = \max(0, 1-\Vert a-a^*\Vert-\langle w^*, x-x^*\rangle) + \varepsilon
$$
By default

- The actions space is $\mathcal A = \{0.0, 0.01, \dots, 0.99\}$
- The context space is $\mathcal X = \{0.0, 0.01, \dots, 0.99\}^5$
- The noise follows a Normal $\mathcal N (0, 0.1)$
- The horizon is $T = 1000$
- $a^*, w^*, x^*$ are generated randomly at initialisation.



