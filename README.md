# ekernels: Explicit Feature Map Approximations of Kernel Functions


This is a small library to help me my research in explicit feature map
approximations of fixed kernel functions (e.g., RBF, Laplace).  

## Problem Statement

Find a finite dimensional function $z: \mathcal X \rightarrow \mathbb R^D$ such that

$$
k(\mathbf x, \mathbf x') \approx \langle z(\mathbf x), z (\mathbf x')  \rangle = \tilde k(\mathbf x, \mathbf x').
$$

For example, if $k$ is the Gaussian Kernel, it is defined as

$$
k(\mathbf x, \mathbf x'; \sigma) = e^{-\frac{\|\mathbf x - \mathbf x'\|^2}{2 \sigma^2} }.
$$

## Types of Feature Maps Approximations

* **deterministic** dense Gaussian quadrature, Taylor series
* **non-deterministic** random fourier features, sparse Gaussian quadrature
* **data-dependent** reweighted RFFs

## References

Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. Advances in neural information processing systems, 20.

Li, K., & Principe, J. C. (2019). No-trick (treat) kernel adaptive filtering using deterministic features. arXiv preprint arXiv:1912.04530.

Dao, T., De Sa, C. M., & Ré, C. (2017). Gaussian quadrature for kernel features. Advances in neural information processing systems, 30.


