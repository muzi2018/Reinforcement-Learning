# Derivation

Bellman equation:

$$
\begin{aligned}
v_\pi(s) & =\mathbb{E}\left[R_{t+1} \mid S_t=s\right]+\gamma \mathbb{E}\left[G_{t+1} \mid S_t=s\right], \\
& =\underbrace{\sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s, a) r}_{\text {mean of immediate rewards }}+\underbrace{\gamma \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)}_{\text {mean of future rewards }} \\
& =\sum_{a \in \mathcal{A}} \pi(a \mid s)\left[\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)\right], \quad \text { for all } s \in \mathcal{S} .
\end{aligned}
$$

matrix-vector form:

$$
v_\pi(s)=r_\pi(s)+\gamma \sum_{s^{\prime} \in \mathcal{S}} p_\pi\left(s^{\prime} \mid s\right) v_\pi\left(s^{\prime}\right)
$$

where

$$
\begin{aligned}
r_\pi(s) & \doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{r \in \mathcal{R}} p(r \mid s, a) r \\
p_\pi\left(s^{\prime} \mid s\right) & \doteq \sum_{a \in \mathcal{A}} \pi(a \mid s) p\left(s^{\prime} \mid s, a\right) .
\end{aligned}
$$

Suppose that the states are indexed as $s_i$ with $i=1, \ldots, n$, where $n=|\mathcal{S}|$. For state $s_i,(2.8)$ can be written as

$$
v_\pi\left(s_i\right)=r_\pi\left(s_i\right)+\gamma \sum_{s_j \in \mathcal{S}} p_\pi\left(s_j \mid s_i\right) v_\pi\left(s_j\right) .
$$

Let $v_\pi=\left[v_\pi\left(s_1\right), \ldots, v_\pi\left(s_n\right)\right]^T \in \mathbb{R}^n, r_\pi=\left[r_\pi\left(s_1\right), \ldots, r_\pi\left(s_n\right)\right]^T \in \mathbb{R}^n$, and $P_\pi \in \mathbb{R}^{n \times n}$ with $\left[P_\pi\right]_{i j}=p_\pi\left(s_j \mid s_i\right)$. Then, the above Bellman equation can be written in the following matrix-vector form:

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi,
$$

where $v_\pi$ is the unknown to be solved, and $r_\pi, P_\pi$ are known.

The example Bellman eqation:

$$
\underbrace{\left[\begin{array}{l}
v_\pi\left(s_1\right) \\
v_\pi\left(s_2\right) \\
v_\pi\left(s_3\right) \\
v_\pi\left(s_4\right)
\end{array}\right]}_{v_\pi}=\underbrace{\left[\begin{array}{l}
r_\pi\left(s_1\right) \\
r_\pi\left(s_2\right) \\
r_\pi\left(s_3\right) \\
r_\pi\left(s_4\right)
\end{array}\right]}_{r_\pi}+\gamma \underbrace{\left[\begin{array}{llll}
p_\pi\left(s_1 \mid s_1\right) & p_\pi\left(s_2 \mid s_1\right) & p_\pi\left(s_3 \mid s_1\right) & p_\pi\left(s_4 \mid s_1\right) \\
p_\pi\left(s_1 \mid s_2\right) & p_\pi\left(s_2 \mid s_2\right) & p_\pi\left(s_3 \mid s_2\right) & p_\pi\left(s_4 \mid s_2\right) \\
p_\pi\left(s_1 \mid s_3\right) & p_\pi\left(s_2 \mid s_3\right) & p_\pi\left(s_3 \mid s_3\right) & p_\pi\left(s_4 \mid s_3\right) \\
p_\pi\left(s_1 \mid s_4\right) & p_\pi\left(s_2 \mid s_4\right) & p_\pi\left(s_3 \mid s_4\right) & p_\pi\left(s_4 \mid s_4\right)
\end{array}\right]}_{P_\pi} \underbrace{\left[\begin{array}{l}
v_\pi\left(s_1\right) \\
v_\pi\left(s_2\right) \\
v_\pi\left(s_3\right) \\
v_\pi\left(s_4\right)
\end{array}\right]}_{v_\pi} .
$$



$$
\left[\begin{array}{c}
v_\pi\left(s_1\right) \\
v_\pi\left(s_2\right) \\
v_\pi\left(s_3\right) \\
v_\pi\left(s_4\right)
\end{array}\right]=\left[\begin{array}{c}
0.5(0)+0.5(-1) \\
1 \\
1 \\
1
\end{array}\right]+\gamma\left[\begin{array}{cccc}
0 & 0.5 & 0.5 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1
\end{array}\right]\left[\begin{array}{l}
v_\pi\left(s_1\right) \\
v_\pi\left(s_2\right) \\
v_\pi\left(s_3\right) \\
v_\pi\left(s_4\right)
\end{array}\right]



$$


## Closed-form solution

premise: $p(r \mid s, a)$ and $p\left(s^{\prime} \mid s, a\right)$ are known

Advantage: useful for theoretical analysis purpose

Disadvantage: not applicable in practice 

Since $v_\pi=r_\pi+\gamma P_\pi v_\pi$ is a simple linear equation, its closed-form solution can be easily obtained as

$$
v_\pi=\left(I-\gamma P_\pi\right)^{-1} r_\pi .
$$



## Iterative solution

numerical algorithm

$v_{k+1}=r_\pi+\gamma P_\pi v_k, \quad k=0,1,2, \ldots$


This algorithm generates a sequence of values $\left\{v_0, v_1, v_2, \ldots\right\}$, where $v_0 \in \mathbb{R}^n$ is an initial guess of $v_\pi$. It holds that

$$
v_k \rightarrow v_\pi=\left(I-\gamma P_\pi\right)^{-1} r_\pi, \quad \text { as } k \rightarrow \infty .
$$
