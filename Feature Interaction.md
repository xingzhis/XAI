# Feature Interaction

##List

- [x] Predictive learning via rule ensembles
- [x] Detecting statistical interactions with additive groves of trees
- [x] Accurate intelligible models with pairwise interactions
Michael Tsang
- [ ] NID：Detecting Statistical Interactions from Neural Network Weights
- [ ] NIT：Neural Interaction Transparency (NIT): Disentangling Learned Interactions for Improved Interpretability
- [ ] Feature Interaction Interpretability: A Case for Explaining Ad-Recommendation Systems via Neural Interaction Detection
- [ ] Learning Global Pairwise Interactions with Bayesian Neural Networks
FM相关
- [ ] FM：Factorization machines
- [ ] FFM：Field-aware Factorization Machines for CTR Prediction
- [ ] DeepFM：DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
- [ ] xDeepFM：xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems
- [ ] Wide & Deep Learning for Recommender Systems
- [ ] Deep & Cross：Deep & Cross Network for Ad Click Predictions
- [ ] AFM：Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks
- [ ] DIN：Deep Interest Network for Click-Through Rate Prediction
- [ ] AutoInt: Automatic feature interaction learning via self-attentive neural networks

##Predictive learning via rule ensembles

> ensemble: rules + linear (robust), penalized, rule by trees (correlated predictor control by edge w/ incentive).
>
> Loss Function: regression and classification
>
> Importance: of a rule and of a predictor.

Interaction: Partial dependence and statistic. Bootstrap to generate a reference distribution.

> $F_{s}\left(\mathbf{x}_{s}\right)=E_{\mathbf{x}_{\backslash s}}\left[F\left(\mathbf{x}_{s}, \mathbf{x}_{\backslash s}\right)\right]$
>
> $H_{j k}^{2}=\sum_{i=1}^{N}\left[\hat{F}_{j k}\left(x_{i j}, x_{i k}\right)-\hat{F}_{j}\left(x_{i j}\right)-\hat{F}_{k}\left(x_{i k}\right)\right]^{2} / \sum_{i=1}^{N} \hat{F}_{j k}^{2}\left(x_{i j}, x_{i k}\right)$
>
> $H_{j}^{2}=\sum_{i=1}^{N}\left[F\left(\mathbf{x}_{i}\right)-\hat{F}_{j}\left(x_{i j}\right)-\hat{F}_{\backslash j}\left(\mathbf{x}_{i \backslash j}\right)\right]^{2} / \sum_{i=1}^{N} F^{2}\left(\mathbf{x}_{i}\right)$
>
> $\begin{aligned} H_{j k l}^{2}=\sum_{i=1}^{N}\left[\hat{F}_{j k l}\left(x_{i j}, x_{i k}, x_{i l}\right)-\hat{F}_{j k}\left(x_{i j}, x_{i k}\right)\right.& \\-\hat{F}_{j l}\left(x_{i j}, x_{i l}\right)-\hat{F}_{k l}\left(x_{i k}, x_{i l}\right)+\hat{F}_{j}\left(x_{i j}\right) & \\\left.+\hat{F}_{k}\left(x_{i k}\right)+\hat{F}_{l}\left(x_{i l}\right)\right]^{2} / \sum_{i=1}^{N} \hat{F}_{j k l}^{2}\left(x_{i j}, x_{i k}, x_{i l}\right) \end{aligned}$



## **Accurate Intelligible Models with Pairwise Interactions**

####GA$^2$M

Greedy, Fisrst fit GAM residual with a subset of pairs $\mathcal S$, then fit the new residula $R$ respectively with each remaining pair, add the best fit pair to $\mathcal S$, and iterate.

####FastInteractionDetection

**FAST**: given bins, first compute histogram and cumulative histogram on target and weight to form a lookup table, then reuse them to calculate sum of target on 4 quadrants for each cut. RSS is easily calculable for bin fctns.



## Detecting statistical interactions with additive groves of trees

$F^*(x)$: Target function

$F(x)$: Highly accurate model

$R_{ij}(x)$: Highly accurate but ban interaction between $x_i$ and $x_j$

RMSE normalized by std: $\begin{array}{c}\operatorname{stRMSE}(F(\mathbf{x}))=\frac{\operatorname{RMSE}(F(\mathbf{x}))}{\operatorname{StD}\left(F^{*}(\mathbf{x})\right)} \\ I_{i j}(F(\mathbf{x}))=\operatorname{stRMSE}(F(\mathbf{x}))-\operatorname{stRMSE}\left(R_{i j}(\mathbf{x})\right)\end{array}$

Significant: $I_{i j}(F(\mathbf{x}))>3 \cdot \operatorname{StD}(\operatorname{stRMSE}(F(\mathbf{x})))$

std from bagging, etc.

1. Highly predictive for interactions — *captures interaction*
2. Highly predictive when restricted by non-interaction pair — *lest pseudo interaction*

Additive Groove of regression trees

**Correlation**: drop features (backward elimination)

1. not hurt performance
2. each var is important and removal drops the performance

max detection of interaction <= performance drop removing one of the variables in the pair

###Comparison with *Predictive learning via rule ensembles*

> the calculation of $H_{ij}^2$ method can be hurt by spurious interaction for sparse regions of datapoint. did not understand what this means. but lets just say it this is better at preventing such situations.
>
> **Did not understand this**

####Difference

1. $F(\mathbf{x})=\hat{a}_{0}+\sum_{k=1}^{K} \hat{a}_{k} r_{k}(\mathbf{x})+\sum_{j=1}^{n} \hat{b}_{j} l_{j}\left(x_{j}\right)$ a regularized regression
2. $F(x)=\sum_{i=1}^KT_i(x)$, optimize RMSE, and limit tree size.



####Pro

~~Statistic: less computation, no need to estimate expectation five times.~~ (But does need to compute RMSE and std.)

Statistic: lest spurious local interactions.

Better than GA2M in that high order interaction is included

#### Con

Model: Hard to capture linear response with tree structure.

without weight could overfit? Bagging solves this. 

 **would all the trees in the restriction procedure lean to one relatively more important variable in the pair?** need we add some penalty or (random) incentive?

###Summary

* defines an intereaction statistic by complete model and interaction-free model. significant if greater than 3*std, std generated by resampling. need to remove correlated.

* uses an additive tree model with a 3-layer algm bagging-growing-backfitting

* restrict an inteaction-free tree groove model by forbidding one of the variables each time and selecting the best for each tree generation. 

  > **Que: why this prevents just using the more important var comparing to a ban-same-brance procedure?**

* beats the *Predictive learning via rule ensembles* statistic in avoiding spurious interaction at sparse regions.

