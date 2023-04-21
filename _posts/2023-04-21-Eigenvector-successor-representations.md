---
layout: post
title: Eigenvector successor representations
---

*This technical note was originally written by me in early 2021 on ways to generalize the successor matrix and enable flexible generalization of reward functions across changing environments. It probably does not make much sense unless you are familiar with the successor representation in RL. I originally planned to explore this a little more but did not have time and now it is unlikely I ever will. I hope somebody might find this interesting or useful.*

The key beneficial property of the successor matrix is that it separates out changes of reward from the transition structure of the environment, allowing for zero-shot generalization when reward is revalued. This arises directly from the fact that the value function is computed simply by the product of the successor matrix $$M$$ and the reward $$R$$, where $$M$$ does not depend on $$R$$,

$$\begin{align}
\label{test_eq}
    V = MR
\end{align}$$

Importantly, to some extent we can go further than this and also get invariance to some transition matrix changes. Specifically, we will see that we can write out an generalization of the successor matrix which is invariant to \emph{changes in eigenvectors} of the transition matrix which leave the eigenvalues unchanged. This is pretty straightforward in actuality. We denote the transition matrix by $$T$$. Then, we note that the successor matrix is defined as \ref{test_eq},

$$\begin{align}
    M = I + \gamma T + \gamma^2 T^2 + \gamma^3 T^3 \dots
\end{align}$$

Importantly, the transition map $$T$$ is an $$S \times S$$ square matrix (since it maps states to states), and thus is always diagonalizable so we apply the eigenvector decomposition to it and write,

$$\begin{align}
    T = Q \Sigma Q^T
\end{align}$$

Where $$\Sigma$$ is a diagonal matrix of eigenvalues of $$T$$ and $$Q$$ is an orthogonal matrix of its eigenvectors. Importantly, the eigendecomposition allows matrix powers to be computed extremely straightforwardly as simply the power of the eigenvalues, i.e.

$$\begin{align}
    T^2 &= (Q \Sigma Q^T)^T (Q \Sigma Q^T) \\
    &= Q \Sigma^T Q^T Q \Sigma Q^T \\
    &= Q \Sigma^T \Sigma Q^T \\
    &= Q \Sigma^2 Q^T
\end{align}$$

Where we just use the orthogonality of Q ($$Q^TQ = QQ^T = I$$). This decomposition thus allows us to write the successor matrix in terms of powers of $$\Sigma$$ separating out the eigenvector matrices $$Q$$ entirely,

$$\begin{align}
    T &= I + \gamma Q \Sigma Q^T + \gamma^2 Q \Sigma^2 Q^T + \gamma^3 Q \Sigma^3 Q^T \dots \\
    &= Q (I+ \gamma \Sigma + \gamma^2 \Sigma^2 + \gamma^3 \Sigma^3 \dots) Q^T \\
    &= Q \tilde{\Sigma} Q^T
\end{align}$$

where $$\tilde{\Sigma} = (I+ \gamma \Sigma + \gamma^2 \Sigma^2 + \gamma^3 \Sigma^3 \dots)$$. This means that the eigenvectors of the transition map are not necessarily accumulated in the value function, only the eigenvalues -- i.e. we can write,

$$\begin{align}
    V = Q \tilde{\Sigma} Q^T R
\end{align}$$

Where the only thing that must be accumulated across trials is $$\tilde{\Sigma}$$. This means that, in effect, we can achieve zero-shot generalization to not just reward using this representation but \emph{any transformation of the dynamics that leaves its eigenvalues unchanged}. As far as I can tell this amounts to similarity transformations -- i.e., we can perform zero-shot generalization of any transformation of the form,

$$\begin{align}
    \tilde{T} = X T X^{-1}
\end{align}$$

where $$X$$ can be any arbitrary transformation matrix. I'm not sure how useful this is in practice (how many normal changes of the transition dynamics take this form?) but nevertheless I think it is a cool and interesting extension of the invariance/generalization properties of the successor representation to some set of changes in the transition dynamics. Perhaps this can be generalized to different sets of matrix transformations of the environmental dynamics. If so, then this could potentially provide a flexible form of zero-shot generalization across environments for RL agents based on the successor representation.
