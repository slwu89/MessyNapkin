# Core Statistics Examples

## Urchins

### EM optimization

As we recall the basic problem with fitting models with random effects is that to find the MLE $\hat{\theta}$ we need to evaluate the intractable integral:

$$
L(\theta) = f_{\theta}(y) = \int f_{\theta}(y,b) db
$$

The EM algorithm avoids messing with that integral and replaces it with another integral that is sometimes more analytically tractable but much more straightforward to approximate. It starts with a parameter guess $\theta{'}$ and the decomposition of the joint likelihood:

$$
\log f_{\theta}(y,b) = \log f_{\theta}(b|y) + \log f_{\theta}(y)
$$

We then take the expectation of this entire expression wrt $f_{\theta^{'}}(b|y)$ (the E step):

$$
E_{b|y,\theta^{'}} \log f_{\theta}(y,b) = E_{b|y,\theta^{'}} \log f_{\theta}(b|y) + \log f_{\theta}(y)
$$

and rewrite as:

$$
Q_{\theta^{'}}(\theta) = P_{\theta^{'}}(\theta) + l(\theta)
$$

Note that $E_{b|y,\theta^{'}} \log f_{\theta}(b|y) = \int \log f_{\theta}(b|y) f_{\theta^{'}}(b|y) db$.

The algorithm is to find (the M step) $\theta^{*} = \text{argmax}_{\theta}Q_{\theta^{'}}(\theta)$ and set $\theta^{'} \leftarrow \theta^{*}$ until convergence.

Above is the basic EM algorithm but Wood uses a higher order Laplace approximation for the E step, which seriously helps in evaluating $E_{b|y,\theta^{'}} \log f_{\theta}(b|y)$. The point is to get a good approximation of $Q_{\theta^{'}}(\theta)$. Let's see how he does it.

#### Approximation of E-step

He does quite a bit of math to arrive at the following:

$$
h_{s} = s \log f_{\theta}(y,b) + \log f_{\theta^{'}}(y,b)
$$

Let $\hat{b}_{s}$ maximize $h_{s}$, and $H_{s} = -\nabla^{2} h_{s}(\hat{b}_{s})$. Then the following expression is the approxiation of $Q_{\theta^{'}}(\theta)$:

$$
\left. \log f_{\theta}(y,\hat{b}) - \frac{1}{2} \frac{1}{ds} \log |H_{s}| \right|_{s=0}
$$

where $\hat{b}$ is the maximizer of $\exp{\log f_{\theta}(y,b)}$.