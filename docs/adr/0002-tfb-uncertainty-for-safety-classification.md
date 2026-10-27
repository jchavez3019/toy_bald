# ADR 0002: Training-Free Bayesianization for Safety-Classification Uncertainty

- Status: Proposed
- Date: 2026-10-26
- Deciders: Project maintainer
- Related: [ADR 0001](0001-code-maintainability.md) (quality gates that apply once this work moves into `src/toy_bald`)

This ADR decides how the project equips its safety classifiers with principled uncertainty estimates. We adopt Training-Free Bayesianization (TFB) [1] of the existing LoRA adapters, which turns a trained, deterministic classifier into an approximate Bayesian one without retraining. From the resulting weight posterior we compute the standard decomposition of predictive uncertainty into aleatoric and epistemic parts, whose epistemic term is the BALD score. The decision covers both the binary classifier and the multi-label classifier (flat and hierarchical variants), is implemented first in a notebook, and is then migrated into the package.

## Context

### The classifiers

Both classifiers share one architecture. A prompt $x$ is tokenized (at most 128 tokens), encoded by `intfloat/multilingual-e5-small` (ME5-small: 12 transformer layers, hidden width 384), mean-pooled over non-padding tokens into an embedding in $\mathbb{R}^{384}$, and mapped by a feed-forward head to a vector of logits. The encoder is adapted with LoRA [3] (rank $r = 16$, $\alpha = 32$, dropout $0.05$) on the query, key, value, and attention-output projections of every layer, i.e. $12 \times 4 = 48$ adapted matrices, each of shape $384 \times 384$. The base encoder weights are frozen; only the LoRA matrices and the head are trained.

The two tasks differ only in the head and labels, all derived from the Aegis 2.0 dataset:

- **Binary.** One logit; the target is $y \in \{0, 1\}$ with $y = 1$ meaning the prompt is unsafe.
- **Multi-label.** Three label groups: the unsafe indicator $y_u \in \{0,1\}$; a needs-caution indicator $y_c \in \{0,1\}$, defined only for safe prompts; and a category vector $\mathbf{y}_{\text{cat}} \in \{0,1\}^{21}$, defined only for unsafe prompts (`Manipulation` folded into `Other`). The *flat* variant has one head with $1 + 1 + 21 = 23$ outputs trained against the concatenated target, with the undefined groups set to zero. The *hierarchical* variant has three heads with $1$, $1$, and $21$ outputs on the shared embedding, trained jointly with each head's loss masked to the prompts for which its labels are defined.

### The problem

A trained network provides a single point estimate $\hat{\theta}$ of its parameters and therefore a single predictive distribution $p(y \mid x, \hat{\theta})$. That distribution expresses how noisy the network believes the label to be at $x$, but it says nothing about how uncertain the network is about $\hat{\theta}$ itself. Three practical consequences motivate this ADR:

1. Fine-tuned networks are typically overconfident: predictions made with confidence $0.9$ are correct less than $90\%$ of the time. For a safety classifier this undermines any policy expressed as a probability threshold (block above $0.9$, escalate to a human between $0.4$ and $0.6$).
2. There is no signal that separates an input that is *inherently ambiguous* from one the model has simply *not learned*. Aegis contains both: its labels aggregate several human annotators, and prompts with annotator disagreement are genuinely ambiguous, whereas prompts unlike the training distribution are unfamiliar.
3. The project's goal is Bayesian Active Learning by Disagreement (BALD) [4, 5], which ranks unlabeled inputs by exactly the second kind of uncertainty. That requires a distribution over parameters.

### Prior work considered

TFB [1] Bayesianizes a trained LoRA adapter after the fact by placing a Gaussian over its $A$ matrix and searching for the largest variance that leaves performance on an anchor dataset essentially unchanged; the authors prove this search is equivalent to KL-regularized variational inference under mild conditions. TokUR [2] applies the same family of low-rank Gaussians to pretrained weights that have no adapter, and decomposes per-token uncertainty with the same entropy decomposition adopted below. BLoB [6] learns the mean and variance of $A$ jointly during fine-tuning and is the parameterization TFB inherits.

## Notation

Vectors are bold lowercase ($\mathbf{h}$), matrices are bold uppercase ($\mathbf{W}$), and scalars are plain. All vectors are column vectors.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $m, n$ | scalars | Output and input width of an adapted linear layer; $m = n = 384$ here |
| $r$ | scalar | LoRA rank; $r = 16 < \min(m, n)$ |
| $s$ | scalar | LoRA scaling $\alpha / r = 2$ |
| $\mathbf{W}_0$ | $m \times n$ | Frozen pretrained weight of a layer |
| $\mathbf{B}$, $\mathbf{A}$ | $m \times r$, $r \times n$ | Trained LoRA matrices (`lora_B`, `lora_A` in PEFT) |
| $\Delta\mathbf{W}$ | $m \times n$ | Weight update $s\,\mathbf{B}\mathbf{A}$ |
| $\mathbf{h}$, $\mathbf{z}$ | $n$, $m$ | Layer input and output, $\mathbf{z} = (\mathbf{W}_0 + \Delta\mathbf{W})\,\mathbf{h}$ |
| $\tilde{\mathbf{B}}$ | $m \times r$ | Scaled factor $s\,\mathbf{B}$, so that $\Delta\mathbf{W} = \tilde{\mathbf{B}}\mathbf{A}$ |
| $\mathbf{U}, \mathbf{d}, \mathbf{V}$ | $m \times r$, $r$, $r \times r$ | Compact SVD $\tilde{\mathbf{B}} = \mathbf{U}\,\mathrm{diag}(\mathbf{d})\,\mathbf{V}^\top$, with $\mathbf{U}^\top\mathbf{U} = \mathbf{I}_r$, $\mathbf{V}^\top\mathbf{V} = \mathbf{V}\mathbf{V}^\top = \mathbf{I}_r$, $d_1 \ge \dots \ge d_r > 0$ |
| $\mathbf{E}$ | $r \times n$ | Random noise added to a factor of $\Delta\mathbf{W}$ |
| $\mathbf{I}_k$ | $k \times k$ | Identity matrix |
| $\mathrm{vec}(\mathbf{X})$ | $pq$ | Columns of $\mathbf{X} \in \mathbb{R}^{p \times q}$ stacked into one vector |
| $\mathbf{X} \otimes \mathbf{Y}$ | $pk \times ql$ | Kronecker product of $\mathbf{X} \in \mathbb{R}^{p\times q}$ and $\mathbf{Y} \in \mathbb{R}^{k\times l}$: the block matrix whose $(i,j)$ block is $X_{ij}\mathbf{Y}$ |
| $\sigma_q$ | scalar | Posterior noise scale shared by all adapted layers |
| $\theta$ | — | All random parameters (the perturbed LoRA factors of all 48 layers) |
| $\phi$ | — | All deterministic parameters ($\mathbf{W}_0$, the orthonormal LoRA factors, biases, layer norms, the head) |
| $q_{\sigma_q}(\theta)$ | — | Variational (approximate) posterior over $\theta$ |
| $\mathbf{g}(x;\theta)$ | $K$ | Head logits; $K = 1$ (binary), $K = 23$ (flat), or $1 + 1 + 21$ split across three heads (hierarchical) |
| $\mathrm{sigm}(t)$ | scalar | Logistic sigmoid $1/(1 + e^{-t})$, applied elementwise to vectors |
| $N$ | scalar | Number of posterior samples $\theta_1, \dots, \theta_N \sim q_{\sigma_q}$ |
| $p_n$ | scalar | $p(y = 1 \mid x, \theta_n) = \mathrm{sigm}(g(x;\theta_n))$ for one binary output |
| $\bar{p}$ | scalar | Monte Carlo predictive mean $\frac{1}{N}\sum_{n} p_n$ |
| $\mathcal{H}_b(p)$ | scalar | Binary entropy $-p\ln p - (1-p)\ln(1-p)$, in nats |
| $\epsilon$ | scalar | Tolerated change in anchor performance when selecting $\sigma_q$ |

## Decision

### 1. Uncertainty is measured with the entropy decomposition, and BALD is its epistemic term

Given a posterior $q(\theta)$, the Bayesian model average (BMA) predictive distribution is

$$
p(y \mid x) = \mathbb{E}_{q(\theta)}\big[p(y \mid x, \theta)\big] \approx \frac{1}{N}\sum_{n=1}^{N} p(y \mid x, \theta_n), \qquad \theta_n \sim q(\theta).
$$

Its entropy, the **total uncertainty**, decomposes exactly into two non-negative terms [2, 4, 5]:

$$
\underbrace{\mathcal{H}\big[p(y \mid x)\big]}_{\text{total (TU)}}
\;=\;
\underbrace{\mathbb{E}_{q(\theta)}\,\mathcal{H}\big[p(y \mid x, \theta)\big]}_{\text{aleatoric (AU)}}
\;+\;
\underbrace{\mathcal{I}(y;\theta \mid x)}_{\text{epistemic (EU)}} .
$$

The two terms correspond to two distinct layers of randomness in the model, and keeping them separate is the reason the decomposition exists:

- **The likelihood $p(y \mid x, \theta)$ carries aleatoric uncertainty.** For a fixed $\theta$, the sigmoid output parameterizes a Bernoulli distribution over the label whose spread varies with $x$; a classifier is therefore already a heteroscedastic (input-dependent) noise model, and $\mathcal{H}[p(y \mid x,\theta)]$ is that particular network's estimate of the label noise at $x$. Aleatoric uncertainty is irreducible: more data or a more expressive model cannot remove ambiguity that is present in the labels themselves, such as Aegis prompts on which annotators disagree.
- **The posterior $q(\theta)$ carries epistemic uncertainty.** Finite data does not determine $\theta$; $q$ describes the set of parameter values that explain the data about equally well. Epistemic uncertainty is reducible with more, or more diverse, data and is largest on inputs unlike the training distribution.

The aleatoric term is an expectation over $q$ because the label noise at $x$ can only be estimated through some network, and we are uncertain which network is correct; it is the label noise *averaged over all plausible networks*, and each term inside the expectation is still a within-network quantity. A deterministic network corresponds to the degenerate posterior $q(\theta) = \delta(\theta - \hat{\theta})$, for which the epistemic term vanishes and only aleatoric uncertainty remains. The heteroscedastic temporal head of the E2E_RAG project is such a case: it predicts an input-dependent variance $s^2_{\hat\theta}(x)$ under a single parameter estimate. Its regression analogue of the decomposition above is the law of total variance,

$$
\mathrm{Var}[y \mid x] = \underbrace{\mathbb{E}_{q(\theta)}\big[s_\theta^2(x)\big]}_{\text{aleatoric}} + \underbrace{\mathrm{Var}_{q(\theta)}\big[\mu_\theta(x)\big]}_{\text{epistemic}},
$$

where $\mu_\theta(x)$ and $s^2_\theta(x)$ are the predicted mean and variance; Bayesianizing that head would add the second term to the first.

**Binary estimators.** For a binary output with samples $p_n$ and mean $\bar{p}$, the Monte Carlo estimators are

$$
\widehat{\mathrm{TU}} = \mathcal{H}_b(\bar{p}), \qquad
\widehat{\mathrm{AU}} = \frac{1}{N}\sum_{n=1}^{N}\mathcal{H}_b(p_n), \qquad
\widehat{\mathrm{EU}} = \mathcal{H}_b(\bar{p}) - \frac{1}{N}\sum_{n=1}^{N}\mathcal{H}_b(p_n).
$$

Here $\mathcal{H}_b(\bar{p})$ is the entropy of the averaged prediction and $\mathcal{H}_b(p_n)$ is the entropy of the prediction made by the $n$-th weight sample. Because $\mathcal{H}_b$ is concave, Jensen's inequality gives $\widehat{\mathrm{EU}} \ge 0$, with equality exactly when all samples agree; all three quantities lie in $[0, \ln 2]$. The epistemic term is large only when individual samples are confident *and* disagree, which is what "disagreement" in BALD refers to:

| Samples $p_n$ | $\bar{p}$ | $\widehat{\mathrm{TU}}$ | $\widehat{\mathrm{AU}}$ | $\widehat{\mathrm{EU}}$ | Reading |
| --- | --- | --- | --- | --- | --- |
| $0.5, 0.5$ | $0.5$ | $0.693$ | $0.693$ | $0$ | Ambiguous input, model is sure it is ambiguous |
| $0.9, 0.1$ | $0.5$ | $0.693$ | $0.325$ | $0.368$ | Confident samples that disagree |

$\widehat{\mathrm{EU}}$ is the BALD acquisition score and the primary uncertainty score in this project. The standard deviation of $\{p_n\}$ is correlated with it and may be plotted, but it is not used for decisions because it lacks the information-theoretic meaning and the clean separation from aleatoric noise.

**Logit variance as a diagnostic.** Let $f_n = f(x; \theta_n)$ be the pre-sigmoid logit of sample $n$, and $\widehat{\mathrm{Var}}_\theta[f] = \frac{1}{N-1}\sum_n (f_n - \bar{f})^2$ its sample variance. A second-order expansion of $\mathcal{H}_b(\sigma(f))$ around $\bar{f}$ gives, for small posterior noise,

$$
\mathrm{EU}(x) \;\approx\; \tfrac{1}{2}\,\bar{p}\,(1-\bar{p})\;\mathrm{Var}_\theta[f(x)] .
$$

The epistemic term is therefore the product of closeness to the decision boundary, $\bar{p}(1-\bar{p})$, and posterior disagreement on the logit. The first factor is exactly what makes EU informative for acquisition: a label is worth requesting only if the samples disagree about the label itself. But the same factor suppresses inputs on which every sample is confident and the samples still disagree strongly about *how* confident, as with logits of 8 and 14. Off-distribution text tends to look like this. $\widehat{\mathrm{Var}}_\theta[f]$ isolates the disagreement factor, so it is reported next to EU in the misclassification and distribution-shift evaluations of Decision 5. It is a diagnostic only, not an acquisition score, because it carries no information-theoretic meaning about the label. In the small-noise regime, every posterior in this ADR has $\mathrm{Var}_\theta[f(x)] \propto \sigma^2$ for each input, so rescaling $\sigma$ changes neither score's ranking.

**Multi-label estimators.** Every output $k$ of a multi-label head is a Bernoulli with its own samples $p_{n,k}$, so the binary estimators apply per output, giving $\widehat{\mathrm{EU}}_k$. Given $\theta$, the outputs are conditionally independent, so $\mathbb{E}_q\,\mathcal{H}[p(\mathbf{y}\mid x,\theta)] = \sum_k \mathbb{E}_q\,\mathcal{H}_b(p_k)$; combined with the subadditivity of entropy, $\mathcal{H}[p(\mathbf{y}\mid x)] \le \sum_k \mathcal{H}_b(\bar{p}_k)$, this yields

$$
\mathcal{I}(\mathbf{y};\theta \mid x) \;\le\; \sum_k \mathcal{I}(y_k;\theta \mid x),
$$

so the per-output sum is an upper bound on the joint epistemic uncertainty and is reported as such. To keep the two multi-label variants comparable and aligned with the binary model, the **primary** score for every model is the epistemic uncertainty of the safe/unsafe decision, $\widehat{\mathrm{EU}}_u$. In addition we report, for the needs-caution output, $\widehat{\mathrm{EU}}_c$ on prompts predicted safe, and for the categories, $\sum_k \widehat{\mathrm{EU}}_{\text{cat},k}$ on prompts predicted unsafe. The hierarchical heads produce conditional probabilities $p(y_{\text{cat},k} = 1 \mid y_u = 1, x, \theta)$, whereas the flat head's outputs approximate marginals; when marginals are needed for comparison, they are formed *per weight sample* as $p_{n,u}\cdot p_{n,\text{cat}\mid u}$ before averaging, so that the marginalization over $\theta$ is done correctly.

### 2. The posterior is TFB's low-rank isotropic Gaussian over the LoRA adapters

Each adapted layer computes $\mathbf{z} = (\mathbf{W}_0 + \tilde{\mathbf{B}}\mathbf{A})\,\mathbf{h}$. Following BLoB [6] and TFB [1], only the $\mathbf{A}$-side factor is random; $\mathbf{W}_0$ and the $\mathbf{B}$-side factor stay deterministic. The posterior mean is the trained weight $\mathbf{W}_0 + \Delta\mathbf{W}$, so Bayesianization never moves the point estimate, and the remaining design question is the covariance.

**Why the noise on $\mathbf{A}$ cannot simply be isotropic.** Suppose the sampled factor is $\mathbf{A} + \mathbf{E}$ with $\mathbf{E} \in \mathbb{R}^{r\times n}$ and independent entries $E_{ij} \sim \mathcal{N}(0, \sigma^2)$. We then have the sampled update $\mathbf{\tilde{B}}(\mathbf{A} + \mathbf{E}) = \mathbf{\tilde{B}A} + \mathbf{\tilde{B}E}$. The random part of the sampled weight is $\tilde{\mathbf{B}}\mathbf{E}$, whose $j$-th column is $\tilde{\mathbf{B}}\mathbf{e}_j$ with $\mathbf{e}_j \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_r)$ the $j$-th column of $\mathbf{E}$. A linear map $\mathbf{L}$ applied to a Gaussian with covariance $\mathbf{\Sigma}$ yields covariance $\mathbf{L}\mathbf{\Sigma}\mathbf{L}^\top$, hence

$$
\mathrm{Cov}(\tilde{\mathbf{B}}\mathbf{e}_j) = \sigma^2\,\tilde{\mathbf{B}}\tilde{\mathbf{B}}^\top = \sigma^2\,\mathbf{U}\,\mathrm{diag}(d_1^2, \dots, d_r^2)\,\mathbf{U}^\top .
$$

Along the $i$-th left singular direction $\mathbf{u}_i$ of $\tilde{\mathbf{B}}$ the weight noise has variance $\sigma^2 d_i^2$: the isotropic noise on $\mathbf{A}$ is stretched by the squared singular values of $\tilde{\mathbf{B}}$. Worse, the posterior is not even well defined as a distribution over weights, because the same update admits infinitely many factorizations $\tilde{\mathbf{B}}\mathbf{A} = (\tilde{\mathbf{B}}\mathbf{S})(\mathbf{S}^{-1}\mathbf{A})$ for any invertible $\mathbf{S} \in \mathbb{R}^{r\times r}$, each producing a different covariance $\sigma^2\tilde{\mathbf{B}}\mathbf{S}\mathbf{S}^\top\tilde{\mathbf{B}}^\top$.

**TFB's construction.** TFB re-factors the update through the SVD of the $\mathbf{B}$-side factor,

$$
\tilde{\mathbf{B}} = \mathbf{U}\,\mathrm{diag}(\mathbf{d})\,\mathbf{V}^\top, \qquad
\mathbf{B}' = \mathbf{U}\,\mathrm{diag}(\mathbf{d}) \in \mathbb{R}^{m\times r}, \qquad
\mathbf{A}' = \mathbf{V}^\top\mathbf{A} \in \mathbb{R}^{r\times n},
$$

which leaves the update unchanged, $\mathbf{B}'\mathbf{A}' = \mathbf{U}\,\mathrm{diag}(\mathbf{d})\,\mathbf{V}^\top\mathbf{A} = \Delta\mathbf{W}$, since $\mathbf{V}$ is orthogonal. It then samples $\mathbf{A}' + \mathbf{E}$ with independent $E_{ij} \sim \mathcal{N}(0, \Omega_{ij}^2)$ and row-dependent standard deviation

$$
\Omega_{ij} = \frac{\sigma_q}{d_i}, \qquad i \in \{1,\dots,r\},\; j \in \{1,\dots,n\}.
$$

The scaling is per row because row $i$ of $\mathbf{A}'$ is exactly what $\mathrm{diag}(\mathbf{d})$ multiplies by $d_i$. The $j$-th column of the weight noise is $\mathbf{B}'\mathbf{e}_j = \mathbf{U}\mathbf{\zeta}_j$ with $\mathbf{\zeta}_j = \mathrm{diag}(\mathbf{d})\,\mathbf{e}_j$, whose $i$-th entry $d_i E_{ij}$ has standard deviation $d_i \cdot \sigma_q / d_i = \sigma_q$. The singular values cancel, $\mathbf{\zeta}_j \sim \mathcal{N}(\mathbf{0}, \sigma_q^2\mathbf{I}_r)$, and therefore

$$
\mathrm{Cov}(\mathbf{U}\mathbf{\zeta}_j) = \sigma_q^2\,\mathbf{U}\mathbf{U}^\top .
$$

The matrix $\mathbf{U}\mathbf{U}^\top \in \mathbb{R}^{m\times m}$ is the orthogonal projector onto the column space of $\tilde{\mathbf{B}}$: the noise has variance $\sigma_q^2$ in every direction inside that $r$-dimensional subspace and zero variance orthogonal to it.

**The full covariance.** Distinct columns $j \neq k$ of the weight noise are built from disjoint, independent entries of $\mathbf{E}$, so their cross-covariance is zero. Stacking the $n$ columns, the covariance of $\mathrm{vec}(\mathbf{W})$ is block diagonal with $n$ identical $m\times m$ blocks, which is precisely what a Kronecker product with an identity expresses:

$$
q_{\sigma_q}\big(\mathrm{vec}(\mathbf{W})\big) = \mathcal{N}\!\Big(\mathrm{vec}(\mathbf{W}_0 + \Delta\mathbf{W}),\; \mathbf{I}_n \otimes \sigma_q^2\,\mathbf{U}\mathbf{U}^\top\Big),
\qquad
\mathbf{I}_n \otimes \sigma_q^2\,\mathbf{U}\mathbf{U}^\top =
\begin{bmatrix}
\sigma_q^2\,\mathbf{U}\mathbf{U}^\top & & \\
& \ddots & \\
& & \sigma_q^2\,\mathbf{U}\mathbf{U}^\top
\end{bmatrix} \in \mathbb{R}^{mn\times mn}.
$$

This is TFB's Theorem 4.1. The paper writes the block as $\sigma_q^2\,\mathrm{diag}(\mathbf{I}_r, \mathbf{0}_{m-r})$, which is the same projector expressed in an orthonormal basis of $\mathbb{R}^m$ whose first $r$ vectors are the columns of $\mathbf{U}$; this ADR uses the basis-free form $\mathbf{U}\mathbf{U}^\top$. The covariance has rank $nr$, so the Gaussian is degenerate: it is isotropic *within* the subspace LoRA can move, not over all of $\mathbb{R}^{mn}$, which no noise of the form $\tilde{\mathbf{B}}\mathbf{E}$ could be. Three properties follow and are the reasons for choosing this family:

1. **One scalar parameter.** The whole posterior across all 48 layers is indexed by a single $\sigma_q$, which reduces posterior selection to a one-dimensional search (Decision 3) and is what makes TFB's equivalence to variational inference provable [1, Theorem 4.2].
2. **Invariance to the factorization.** Re-factoring $\tilde{\mathbf{B}} \mapsto \tilde{\mathbf{B}}\mathbf{S}$ leaves the column space, and hence $\mathbf{U}\mathbf{U}^\top$, unchanged, so every factorization of the same $\Delta\mathbf{W}$ yields the same posterior.
3. **Compact storage.** Beyond the adapter itself, each layer needs only $\mathbf{d} \in \mathbb{R}^r$ (or nothing, in the implementation below), instead of a full $r\times n$ standard-deviation matrix.

**Implementation parameterization.** Absorbing $\mathrm{diag}(\mathbf{d})$ into the $\mathbf{A}$-side factor gives an equivalent and numerically simpler form. Define

$$
\mathbf{B}'' = \mathbf{U} \in \mathbb{R}^{m\times r}, \qquad
\mathbf{A}'' = \mathrm{diag}(\mathbf{d})\,\mathbf{V}^\top\mathbf{A} = \mathbf{U}^\top\Delta\mathbf{W} \in \mathbb{R}^{r\times n}, \qquad
\mathbf{W} = \mathbf{W}_0 + \mathbf{B}''\big(\mathbf{A}'' + \mathbf{Z}\big),
$$

with $\mathbf{Z} \in \mathbb{R}^{r\times n}$ and independent $Z_{ij} \sim \mathcal{N}(0, \sigma_q^2)$. Since $\mathbf{U}\mathbf{U}^\top\Delta\mathbf{W} = \Delta\mathbf{W}$, the mean is unchanged, and $\mathbf{U}\mathbf{Z}$ is exactly the noise $\mathbf{U}\mathbf{\zeta}_j$ derived above. This form also makes the role of the SVD transparent: it replaces the $\mathbf{B}$-side factor with one that has orthonormal columns, and isotropic noise multiplied by a matrix with orthonormal columns stays isotropic. It requires no division by $d_i$. Directions with negligible singular values carry essentially no learned update and have numerically unstable singular vectors, so only the $r_{\text{eff}} = |\{i : d_i > \tau\,d_1\}|$ leading directions are kept, with $\tau = 10^{-4}$; TFB itself assumes $\tilde{\mathbf{B}}$ has full column rank.

**Scope.** The posterior factorizes over layers, $q_{\sigma_q}(\theta) = \prod_{\ell=1}^{48} q_{\sigma_q}\big(\mathrm{vec}(\mathbf{W}^{(\ell)})\big)$, with one shared $\sigma_q$, as in [1]. In phase 1 everything else, including the feed-forward head, is deterministic. As an optional phase-2 variant, the head's weight matrices $\mathbf{W}^{\text{head}}_k$ receive full-rank isotropic noise $\mathbf{W}^{\text{head}}_k + \sigma_h\,\rho_k\,\mathbf{\Xi}_k$, where $\mathbf{\Xi}_k$ has independent standard-normal entries, $\rho_k$ is the root-mean-square of the entries of $\mathbf{W}^{\text{head}}_k$, and $\sigma_h$ is a second scalar searched after $\sigma_q$ is fixed. Full rank is preferred over TokUR's low-rank form for the head because its matrices ($384\times 512$ and $512\times 512$) are small enough that low rank saves nothing, and the per-layer factor $\rho_k$ is needed because, unlike the LoRA updates, the head's layers have different weight scales. This variant is evaluated as an ablation; it is also the cheaper route to many samples, because the encoder output can be reused across head samples. Concretely:

- **Which weights.** All five weight matrices of the binary head are perturbed: $512\times 384$, three of $512\times 512$, and the $1\times 512$ output layer. The biases stay deterministic, mirroring the LoRA side, where only weights are random.
- **Search.** $\sigma_h$ is searched with the LoRA posterior fixed at $\sigma_q^\star$. It is the largest value with $|\mathcal{L}(\sigma_q^\star, \sigma_h) - \mathcal{L}(\sigma_q^\star, 0)| \le \epsilon$, using the same anchor set, pseudo-labels, $\epsilon$ and bracketing-plus-bisection procedure as Decision 3. Measuring the change against $\mathcal{L}(0)$ instead would leave no budget, because the bisection places $\sigma_q^\star$ where the LoRA noise alone already changes the loss by $\epsilon$. Each noise source therefore receives its own budget of $\epsilon$, and in the small-noise regime the two changes add, since the cross terms of independent zero-mean noise vanish in expectation.
- **Sampling.** Head sample $n$ uses the antithetic sign of LoRA sample $n$, so the joint noise $(\mathbf{Z}_n, \mathbf{\Xi}_n)$ is antithetic and the odd-order terms cancel jointly. Head and LoRA noise are drawn from disjoint seed ranges, so they are independent. Samples are applied functionally to cached encoder outputs, which leaves the trained head untouched.
- **Comparison.** Three posteriors are evaluated under Decision 5: LoRA only ($\sigma_q^\star$, 0), head only (0, $\sigma_h^\star$), and LoRA + head ($\sigma_q^\star$, $\sigma_h^\star$).

### 3. The noise scale is the largest one that leaves anchor-set NLL essentially unchanged

Following [1, Eq. 8 and Algorithm 1], $\sigma_q$ is the solution of

$$
\sigma_q^\star = \max_{\sigma_q \ge 0}\; \sigma_q
\quad \text{s.t.} \quad
\big|\,\mathcal{L}(\sigma_q) - \mathcal{L}(0)\,\big| \le \epsilon,
\qquad
\mathcal{L}(\sigma_q) = -\frac{1}{|\mathcal{D}_a|}\sum_{(x, \hat{y}) \in \mathcal{D}_a} \mathbb{E}_{q_{\sigma_q}(\theta)}\big[\ln p(\hat{y} \mid x, \theta)\big],
$$

the largest posterior variance whose expected negative log-likelihood on an anchor set $\mathcal{D}_a$ stays within $\epsilon$ of the deterministic model's. $\mathcal{L}(0)$ is the deterministic NLL. Note that $\mathcal{L}$ is the expected NLL under $q$, as in [1, Assumption 4.1], not the NLL of the BMA predictive used for evaluation in Decision 5. The concrete choices are:

- **Anchor set.** $|\mathcal{D}_a| = 500$ records drawn uniformly, with a fixed seed, from the training subset, as in [1]. Using training data requires no separate calibration split.
- **Pseudo-labels.** $\hat{y}$ is the deterministic model's own prediction (threshold $0.5$), not the ground truth. This is the protocol of [1], and it makes the procedure label-free, which is what the later active-learning work on unlabeled pools needs.
- **Likelihood.** Unweighted binary cross-entropy. The class weights used during training are deliberately excluded, because they would distort the likelihood that the constraint is meant to preserve. For the flat multi-label model the NLL sums over all 23 outputs; for the hierarchical model it sums head 1 over all anchors, head 2 over anchors pseudo-labeled safe, and head 3 over anchors pseudo-labeled unsafe, mirroring the training masks.
- **Tolerance.** $\epsilon = 0.003\,\mathcal{L}(0)$, the $0.3\%$ relative change used in [1]. An accuracy-based constraint ($1\%$ relative drop) is run as an ablation.
- **Search.** Bracketing first, then bisection. The bracket starts at $\sigma_{\max} = 10^{-3}$ and doubles until the constraint is violated, because the scale that suits ME5-small need not match the range $[0.001, 0.015]$ reported for Llama-3.1-8B. Eight bisection steps then refine $\sigma_q^\star$. Bisection assumes $\mathcal{L}$ is non-decreasing in $\sigma_q$ near zero, which the local-convexity assumption of [1, Theorem 4.2] supports; the full curve $\mathcal{L}(\sigma_q)$ is plotted to confirm it.
- **Common random numbers.** The expectation in $\mathcal{L}$ is estimated with $N_s = 10$ samples. Standard-normal matrices $\mathbf{Z}^{(\ell)}_n / \sigma_q$ are drawn once per layer and sample, and reused, rescaled, at every candidate $\sigma_q$. Without this, Monte Carlo noise between candidates can exceed $\epsilon$ and break the bisection.
- **Antithetic pairs.** Common random numbers alone leave a first-order error. Write the per-anchor log-likelihood as a function $g(\mathbf{z})$ of the stacked standard-normal noise $\mathbf{z}$, so that a sample at scale $\sigma_q$ evaluates $g(\sigma_q\mathbf{z})$. Expanding around $\mathbf{z} = \mathbf{0}$ gives $\frac{1}{N_s}\sum_n g(\sigma_q\mathbf{z}_n) - g(\mathbf{0}) = \sigma_q\,\nabla g^\top\bar{\mathbf{z}} + \frac{\sigma_q^2}{2N_s}\sum_n \mathbf{z}_n^\top\nabla^2 g\,\mathbf{z}_n + O(\sigma_q^3)$, where $\bar{\mathbf{z}} = \frac{1}{N_s}\sum_n\mathbf{z}_n$. The true expectation has no linear term, because $\mathbb{E}[\mathbf{z}] = \mathbf{0}$, but a finite sample has $\bar{\mathbf{z}} \neq \mathbf{0}$. Reusing the same $\mathbf{z}_n$ at every $\sigma_q$ freezes that error rather than removing it. It then dominates $|\mathcal{L}(\sigma_q) - \mathcal{L}(0)|$ at small $\sigma_q$, can have either sign, and makes the curve non-monotonic, which breaks the bisection's assumption. Samples are therefore drawn in pairs $(\mathbf{Z}_k, -\mathbf{Z}_k)$. Each sample is still an exact draw from $q_{\sigma_q}$, but $\bar{\mathbf{z}} = \mathbf{0}$ holds exactly and every odd-order term cancels. The estimated change is then the curvature term plus $O(\sigma_q^4)$, as the constraint intends. The same pairing is used at inference, where it also reduces the variance of $\bar{p}$. Sample counts must be even.

### 4. Predictions average fifty posterior samples

At inference the model draws $N = 50$ samples $\theta_n \sim q_{\sigma_q^\star}$, in 25 antithetic pairs. Reference [1] uses $N = 10$. On the binary classifier, however, the mean epistemic estimate was still rising between $N = 10$ and $N = 50$, by about 60%, because the plug-in mutual-information estimator is biased low at small $N$. The search in Decision 3 keeps $N_s = 10$, since there only the difference $\mathcal{L}(\sigma_q) - \mathcal{L}(0)$ matters. Each sample materializes perturbed adapters by writing $\mathbf{B}''$ and $\mathbf{A}'' + \mathbf{Z}_n$ into the PEFT `lora_B` and `lora_A` weights (with $\mathbf{B}''$ divided by $s$ to cancel PEFT's scaling). It then runs the dataset through the model with that fixed $\theta_n$ and restores the trained weights afterwards. The model is in evaluation mode throughout, so that LoRA dropout does not add a second, unintended source of randomness. The resulting per-sample probabilities feed both the BMA prediction $\bar{p}$ and the estimators of Decision 1. Inference cost is $N$ encoder passes per input.

### 5. Evaluation protocol

Every metric is reported for the deterministic model ($\sigma_q = 0$) and the Bayesianized model ($\sigma_q = \sigma_q^\star$) on the test split, using ground-truth labels.

**Predictive quality.** Accuracy, BMA negative log-likelihood $-\frac{1}{|\mathcal{D}|}\sum \ln p(y \mid x)$, and expected calibration error (ECE). A classifier is calibrated if, among predictions made with confidence $c$, a fraction $c$ is correct. ECE measures the deviation from this ideal. Each test example $i$ gets a predicted class $\hat{y}_i$ and confidence $c_i$; for a binary output, $\hat{y}_i = \mathbb{1}[\bar{p}_i \ge 0.5]$ and $c_i = \max(\bar{p}_i, 1 - \bar{p}_i)$. The interval $[0,1]$ is split into $K = 15$ equal-width bins, with $\mathcal{B}_k$ the examples whose confidence falls in bin $k$, and

$$
\mathrm{ECE} = \sum_{k=1}^{K}\frac{|\mathcal{B}_k|}{|\mathcal{D}|}\,\Big|\,\mathrm{acc}(\mathcal{B}_k) - \mathrm{conf}(\mathcal{B}_k)\,\Big|,
\qquad
\mathrm{acc}(\mathcal{B}_k) = \frac{1}{|\mathcal{B}_k|}\sum_{i\in\mathcal{B}_k}\mathbb{1}[\hat{y}_i = y_i],
\quad
\mathrm{conf}(\mathcal{B}_k) = \frac{1}{|\mathcal{B}_k|}\sum_{i\in\mathcal{B}_k} c_i .
$$

ECE is the weighted average gap between the reliability diagram and its diagonal. It is the headline metric of [1], because averaging over the posterior mainly corrects overconfidence. It is always reported next to accuracy and NLL, because it depends on the binning and can be trivially minimized by a useless predictor that always outputs the base rate. For multi-label models, ECE is computed per output and also pooled over all (example, output) pairs on which the output is defined.

**Uncertainty quality.** None of the following needs the uncertainty values themselves to be "correct"; they test whether the scores rank inputs usefully.

- *Misclassification detection* [2]: AUROC and AUPRC of $\widehat{\mathrm{TU}}$, $\widehat{\mathrm{AU}}$, $\widehat{\mathrm{EU}}$, and the logit variance $\widehat{\mathrm{Var}}_\theta[f]$ (Decision 1) as scores for flagging test examples the BMA prediction gets wrong.
- *Distribution shift*: the distributions of $\widehat{\mathrm{EU}}_u$ and $\widehat{\mathrm{Var}}_\theta[f_u]$ on non-safety text, using the locally cached `wikitext-2` and `no_robots` corpora filtered to 128 tokens, against the Aegis test split, each summarized by an AUROC. Epistemic uncertainty should be clearly higher off-distribution.
- *Posterior comparison* (phase 2): the predictive-quality, misclassification, distribution-shift and aleatoric evaluations above, repeated for the LoRA-only, head-only, and LoRA + head posteriors of Decision 2.
- *Aleatoric sanity check*: the distribution of $\widehat{\mathrm{AU}}_u$ on Aegis prompts with annotator disagreement, against the rest. These are safe prompts whose `violated_categories` list unsafe categories, and prompts carrying `Needs Caution`. Aleatoric, not necessarily epistemic, uncertainty should be higher on them.
- *Sensitivity*: accuracy, NLL, ECE, and mean $\widehat{\mathrm{EU}}$ as functions of $\sigma_q$ and of $N$.

Epistemic uncertainty on training versus test prompts is reported for completeness but is **not** a success criterion. Both splits are in-distribution, and $\epsilon$ is chosen precisely to keep the posterior tight around the trained weights, so a small or absent gap is expected and uninformative.

### 6. Binary and multi-label specifics

| Aspect | Binary | Flat multi-label | Hierarchical multi-label |
| --- | --- | --- | --- |
| Outputs | $p_u$ | $p_u$, $p_c$, $\mathbf{p}_{\text{cat}} \in [0,1]^{21}$ from one head | $p_u$; $p_{c\mid\text{safe}}$; $\mathbf{p}_{\text{cat}\mid\text{unsafe}}$ from three heads |
| Search NLL (Decision 3) | BCE on $p_u$ | BCE summed over 23 outputs | Masked sum over heads by pseudo-labeled branch |
| Primary score | $\widehat{\mathrm{EU}}_u$ | $\widehat{\mathrm{EU}}_u$ | $\widehat{\mathrm{EU}}_u$ |
| Secondary scores | — | $\widehat{\mathrm{EU}}_c$, $\sum_k\widehat{\mathrm{EU}}_{\text{cat},k}$ | Same, on the predicted branch, using conditional probabilities |
| Calibration | ECE on $p_u$ | Per-output and pooled ECE | Per-head; category ECE on true-unsafe prompts |

The posterior itself (Decision 2) and the inference procedure (Decision 4) are identical across all three models, because they act only on the shared encoder's adapters.

### 7. Implementation plan

**Prerequisite.** TFB operates on a trained adapter, so the training notebooks must persist the PEFT adapter, the head's `state_dict`, and the configuration needed to rebuild the model (model ID, LoRA settings, `MAX_LEN`, and for multi-label the category vocabulary and variant). Checkpointing during training is tracked separately; this ADR only depends on a final checkpoint existing.

**Phase 1: notebook.** A new notebook, `notebooks/tfb_uncertainty.ipynb`, loads a checkpoint of any of the three models and proceeds in order:

1. Reparameterize every adapted layer into $(\mathbf{B}'', \mathbf{A}'')$, assert $\mathbf{B}''\mathbf{A}'' = \Delta\mathbf{W}$, and record each layer's spectrum $\mathbf{d}$ and $r_{\text{eff}}$.
2. Search for $\sigma_q^\star$, plotting $\mathcal{L}(\sigma_q)$.
3. Compute per-sample probabilities on the train, test, and out-of-distribution sets, then the estimators of Decision 1.
4. Produce the evaluations of Decision 5.

**Phase 2: head variant.** Add the optional full-rank head noise of Decision 2 and compare it against phase 1 under the same protocol.

**Phase 3: package.** Migrate the logic into `src/toy_bald/uncertainty/` under the gates of ADR 0001:

- `tfb.py`: reparameterization and a context manager that applies and restores one posterior sample.
- `search.py`: bracketing and bisection with common random numbers.
- `decomposition.py`: the TU/AU/EU estimators.
- `calibration.py`: ECE and NLL.

Tests use small random matrices on the CPU and cover the following, with GPU and end-to-end checks behind the `gpu` and `slow` markers:

- $\mathbf{B}''\mathbf{A}'' = \Delta\mathbf{W}$ to numerical precision.
- $\sigma_q = 0$ reproduces the deterministic logits exactly.
- A Monte Carlo estimate of $\mathrm{Cov}(\mathrm{vec}(\mathbf{W}))$ converges to $\mathbf{I}_n \otimes \sigma_q^2\mathbf{U}\mathbf{U}^\top$.
- The posterior is invariant under $\tilde{\mathbf{B}}\mapsto\tilde{\mathbf{B}}\mathbf{S}$, $\mathbf{A}\mapsto\mathbf{S}^{-1}\mathbf{A}$.
- $0 \le \widehat{\mathrm{EU}} \le \widehat{\mathrm{TU}} \le \ln 2$, as a Hypothesis property test.
- ECE is zero for a perfectly calibrated synthetic predictor.
- Applying and restoring a sample leaves the adapter weights bit-identical.

The BALD active-learning experiment, which repeatedly acquires the highest-$\widehat{\mathrm{EU}}_u$ prompts from the unused Aegis training pool, will be specified in a follow-up ADR that builds on this one.

## Alternatives considered

- **BLoB [6] and Bayes-by-Backprop.** These learn a mean and variance per parameter by maximizing the evidence lower bound. They are more expressive, but they require retraining with a modified objective and roughly double the trainable parameters, while TFB reuses the existing checkpoints unchanged.
- **Monte Carlo dropout, deep ensembles, Laplace-LoRA.** All are viable baselines. Dropout and ensembles require retraining or several training runs, and Laplace requires curvature estimates by backpropagation. In [1] they are generally matched or outperformed by TFB on calibration.
- **Constant standard deviation on $\mathbf{A}$ ("C-STD" in [1]).** This is the naive scheme analysed in Decision 2: anisotropic in weight space and dependent on the arbitrary factorization of $\Delta\mathbf{W}$. It is empirically worse on accuracy in [1].
- **Full-rank isotropic noise on $\Delta\mathbf{W}$ ("FR" in [1]).** It is competitive with TFB, but it perturbs directions the adapter never learned and costs $O(mn)$ noise per layer instead of $O(rn)$. It is retained only for the small head, where the cost argument does not apply.
- **TokUR's perturbation of pretrained weights [2].** This is $\mathbf{W}_0 + \mathbf{U}_{r'}\mathbf{G}^\top$, with $\mathbf{U}_{r'} \in \mathbb{R}^{m\times r'}$ the top $r'$ left singular vectors of $\mathbf{W}_0$ and $\mathbf{G} \in \mathbb{R}^{n\times r'}$ independent $\mathcal{N}(0,\sigma^2)$ entries. It is the same low-rank isotropic family built from $\mathbf{W}_0$ instead of $\mathbf{B}$, designed for models without adapters, and it uses a fixed $\sigma$ rather than a searched one. Our models have adapters, which define the subspace the fine-tuning actually moved, so TFB is the better fit.
- **Standard deviation of sampled probabilities as the score.** It is simpler, but it does not separate aleatoric from epistemic uncertainty and is not the BALD criterion.

## Consequences

The project gains calibrated BMA predictions and a principled, label-free epistemic score for all three classifiers. No training procedure changes, and none of the existing checkpoints are invalidated.

- Positive: post-hoc and training-free; one scalar hyperparameter, chosen by a documented criterion; identical machinery for the binary and both multi-label models; directly provides the BALD score needed for active learning.
- Negative: inference is $N$ times more expensive; $\sigma_q^\star$ is only as good as the anchor set represents the training distribution; the low-rank isotropic family cannot express correlations between directions or layers. Epistemic estimates are small in absolute terms by construction, so rankings matter more than raw values.
- Follow-up: checkpoint persistence in the training notebooks; the phase-2 head variant; the active-learning ADR; migration into `src/toy_bald/uncertainty/`.

## References

1. H. Shi, Y. Wang, L. Han, H. Zhang, H. Wang. *Training-Free Bayesianization for Low-Rank Adapters of Large Language Models.* NeurIPS 2025. arXiv:2412.05723.
2. T. Zhang, H. Shi, Y. Wang, et al. *TokUR: Token-Level Uncertainty Estimation for Large Language Model Reasoning.* ICLR 2026. arXiv:2505.11737.
3. E. J. Hu, Y. Shen, P. Wallis, et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
4. N. Houlsby, F. Huszár, Z. Ghahramani, M. Lengyel. *Bayesian Active Learning for Classification and Preference Learning.* arXiv:1112.5745, 2011.
5. Y. Gal, R. Islam, Z. Ghahramani. *Deep Bayesian Active Learning with Image Data.* ICML 2017.
6. Y. Wang, H. Shi, L. Han, D. Metaxas, H. Wang. *BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models.* NeurIPS 2024.
