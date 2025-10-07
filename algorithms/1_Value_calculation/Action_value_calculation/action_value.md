# Derivation

From state value to action value

The action value of a state-action pair $(s, a)$ is defined as

$$
q_\pi(s, a) \doteq \mathbb{E}\left[G_t \mid S_t=s, A_t=a\right]
$$

the relationship between action values and state values

1.state value is the expectation of the action values associated with that state, it show how to get state values from action values

$\underbrace{\mathbb{E}\left[G_t \mid S_t=s\right]}_{v_\pi(s)}=\sum_{a \in \mathcal{A}} \underbrace{\mathbb{E}\left[G_t \mid S_t=s, A_t=a\right]}_{q_\pi(s, a)} \pi(a \mid s)$.

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$

2.it show how to get action values from state values, action values consists of two terms. The first term is the mean

of the immediate rewards, and the second term is the mean of the future rewards.

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left[\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)\right]$

$q_\pi(s, a)=\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)$


## Bellman equation

$v_\pi(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s) q_\pi(s, a)$.

$q_\pi(s, a)=\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) v_\pi\left(s^{\prime}\right)$

=

$q_\pi(s, a)=\sum_{r \in \mathcal{R}} p(r \mid s, a) r+\gamma \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in \mathcal{A}\left(s^{\prime}\right)} \pi\left(a^{\prime} \mid s^{\prime}\right) q_\pi\left(s^{\prime}, a^{\prime}\right)$,

### Matrix-vector

$q_\pi=\tilde{r}+\gamma P \Pi q_\pi$

where:

$q_\pi=\left[\begin{array}{c}q_\pi\left(s_1, a_1\right) \\ q_\pi\left(s_1, a_2\right) \\ \vdots \\ q_\pi\left(s_1, a_m\right) \\ q_\pi\left(s_2, a_1\right) \\ q_\pi\left(s_2, a_2\right) \\ \vdots \\ q_\pi\left(s_2, a_m\right) \\ \vdots \\ q_\pi\left(s_n, a_1\right) \\ q_\pi\left(s_n, a_2\right) \\ \vdots \\ q_\pi\left(s_n, a_m\right)\end{array}\right]$

$\tilde{r}=\left[\begin{array}{c}\tilde{r}\left(s_1, a_1\right) \\ \tilde{r}\left(s_1, a_2\right) \\ \vdots \\ \tilde{r}\left(s_1, a_n\right) \\ \tilde{r}\left(s_2, a_1\right) \\ \tilde{r}\left(s_2, a_2\right) \\ \vdots \\ \tilde{r}\left(s_n, a_m\right)\end{array}\right]$

$\tilde{r}\left(s_1, a_1\right)$$=\sum_{r \in R} p\left(r \mid s_i, a_j\right) r$


$$
P=\left[\begin{array}{c}
P^{(1)} \\
P^{(2)} \\
\vdots \\
P^{(n\times m)}
\end{array}\right]
$$

Where each row $P^{(s_i \times a_j)}$ for state-action $(s_i,a_j)$ is defined as:

$$
P^{(s_i \times a_j)}=\left[\begin{array}{cccc}
p\left(s_1 \mid s_i, a_j\right) & p\left(s_2 \mid s_i, a_j\right) & \cdots & p\left(s_n \mid s_i, a_j\right) 
\end{array}\right]
$$

$\Pi=\left[\begin{array}{cccc}\pi\left(a_1 \mid s_1\right) & \cdots & \pi\left(a_m \mid s_1\right) & 0 & \cdots & 0\\ 0 & \cdots & 0 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots \\ 0 & \cdots & 0 & \pi\left(a_1 \mid s_n\right) &\cdots & \pi(a_m \mid s_n))\end{array}\right]$

#### Example

2 states and 2 actions:


$q_\pi=\left[\begin{array}{l}r\left(s_1, a_1\right) \\ r\left(s_1, a_2\right) \\ r\left(s_2, a_1\right) \\ r\left(s_2, a_2\right)\end{array}\right]+\gamma\left[\begin{array}{ll}p\left(s_1 \mid s_1, a_1\right) & p\left(s_2 \mid s_1, a_1\right) \\ p\left(s_1 \mid s_1, a_2\right) & p\left(s_2 \mid s_1, a_2\right) \\ p\left(s_1 \mid s_2, a_1\right) & p\left(s_2 \mid s_2, a_1\right) \\ p\left(s_1 \mid s_2, a_2\right) & p\left(s_2 \mid s_2, a_2\right)\end{array}\right]\left[\begin{array}{ccc}\pi\left(a_1 \mid s_1\right) & \pi\left(a_2 \mid s_1\right) & 0 \\ 0 & 0 & \pi\left(a_1 \mid s_2\right) \\ 0\left(a_2 \mid s_2\right)\end{array}\right]\left[\begin{array}{l}q_\pi\left(s_1, a_1\right) \\ q_\pi\left(s_1, a_2\right) \\ q_\pi\left(s_2, a_1\right) \\ q_\pi\left(s_2, a_2\right)\end{array}\right]$
