# ML-Based Monetary Policy Scenario Analysis and Decision-Support Prototype

## Project goal

The goal of this project is to build a model that can support **monetary policy decisions**. I consider the problem from the point of view of a central bank: given the current state of the economy, the task is to choose the policy rate that helps move inflation toward its target.

The project uses quarterly US macroeconomic data: `FEDFUNDS`, `DTWEXBGS`, `GDPC1`, `CPIAUCSL`, and `RTWEXBGS`. The full dataset is available here: https://github.com/stepan-sapunkov/financial-ml-notes/blob/main/drafts/MP/US_dataQ.xlsx.

For a fair final test, all model development was performed on a reduced dataset that does not contain the final holdout period: https://github.com/stepan-sapunkov/financial-ml-notes/blob/main/drafts/MP/train.csv.

The original time series are shown below.

![Original quarterly US macroeconomic data](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/data.png)

A standard way to solve this problem would be to use models such as **BVAR, SVAR, or DSGE**. In this project I intentionally follow a different direction. The main idea is to combine machine learning with several ideas that are common in quantitative finance. The resulting pipeline consists of five main steps.

---

## Step 1. EDA, transformations, and feature generation

The first step is exploratory data analysis and construction of the feature space.

Because most macroeconomic variables are non-stationary in levels, I transformed the series before fitting the models. For all variables except the policy rate, I use log differences:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\Delta\log(x_t)=\log(x_t)-\log(x_{t-1})" />
</p>

For `FEDFUNDS`, I use the first difference:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5CDelta%20i_t%3Di_t-i_%7Bt-1%7D" />
</p>

where \(i_t\) denotes the policy rate.

This transformation is useful not only from the time-series point of view. It is also important for tree-based ML models. Trees are generally poor at extrapolating outside the range observed during training. If a model is trained directly on interest-rate levels, a future rate level that was never observed during training may result in an almost constant or otherwise unreliable prediction. Working with changes reduces this problem.

### Symbolic feature generation with PySR

Instead of restricting feature engineering to standard transformations such as lags, differences, products, and ratios, I also used **symbolic regression**. For this purpose I used [PySR](https://github.com/astroautomata/PySR).

PySR searches for mathematical expressions that explain a target variable. Instead of estimating parameters inside one fixed equation, it searches over many possible equations constructed from variables, constants, and mathematical operators. Candidate expressions are generated, modified, combined, simplified, and evaluated. In this way, the algorithm searches for equations that provide a reasonable trade-off between predictive quality and mathematical complexity.

In this project PySR is used mainly as a **feature generator**. The discovered equations are not interpreted as economic laws. Instead, useful symbolic transformations are added to the ML feature space.

For example, one of the generated transformations was

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?z_t=\mathrm{FEDFUNDS}_{t-1}\left(\frac{\mathrm{CPIAUCSL}_{t-1}}{0.040583}%2B0.58368\right)" />
</p>

This is the type of transformation that would be difficult to propose manually during ordinary feature engineering.

![Example of a PySR-generated feature](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/scatter.png)

An important restriction is that **PySR was applied only to the training data**. The final holdout sample was not used to discover symbolic expressions. Otherwise, information from the test period could indirectly enter the model through the feature-generation stage even if the final estimator itself were trained only on the training sample.

---

## Step 2. Recovering the unexpected component of the policy rate

The next problem is that the Federal Funds Rate cannot simply be treated as an exogenous variable. The Federal Reserve changes the rate in response to inflation and other macroeconomic conditions, while the interest rate itself affects future inflation. Therefore, there is a two-way relationship:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmathrm%7BInflation%7D%5Clongrightarrow%5Cmathrm%7BPolicy%5C%20Rate%7D%5Clongrightarrow%5Cmathrm%7BFuture%5C%20Inflation%7D" />
</p>

Using the observed policy rate directly inside the inflation model can therefore create an endogeneity problem.

A classical solution would be to identify structural shocks using, for example, an SVAR. I intentionally did not follow this approach. Instead, I first estimated the **predictable component of the policy rate** using ML.

Let

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?i_t=g_h(X_t)%2Bu_t^{i}" />
</p>

Here, \(i_t\) is the observed Federal Funds Rate, \(X_t\) represents the available macroeconomic information, \(g_h(\cdot)\) describes the predictable part of the policy rate, and \(u_t^i\) is the remaining unexpected component.

After estimating the policy rule, this unexpected part is approximated by

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{u}_t^{i}=i_t-\hat{g}_h(X_t)" />
</p>

In simple terms, the first model estimates what policy rate would normally be expected given the current state of the economy. The difference between the observed rate and this prediction is then treated as a proxy for an unexpected policy movement.

This residual should not automatically be interpreted as a structurally identified monetary policy shock. It is better understood as an ML-based proxy for the unexpected part of the policy rate. A stronger causal interpretation would require additional economic identification assumptions.

To avoid optimistic in-sample estimates, the rate trajectories used in the next stages are based on **out-of-sample (OOS) predictions**.

![OOF policy-rate predictions](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/rate.png)

---

## Step 3. Estimating the inflation response

After recovering the policy-rate component, the next step is to estimate the inflation function. For every forecast horizon \(h\), the general structure can be written as

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{\pi}_{t%2Bh}=f_h\left(X_t,\hat{u}_t^{i}\right)" />
</p>

where \(\hat{\pi}_{t+h}\) is predicted inflation \(h\) quarters ahead, \(X_t\) contains the macroeconomic features, \(\hat{u}_t^i\) represents the recovered policy-rate component, and \(f_h(\cdot)\) is the corresponding ML model.

The forecasting problem is solved **separately for each horizon**. This is important because variables that are useful for predicting inflation one quarter ahead do not necessarily have to be equally useful eight quarters ahead. Therefore, each horizon can have its own feature set, PySR transformations, policy-rate specification, inflation model, and final ensemble weights.

PySR feature generation for the inflation models was again performed strictly on the training sample. No information from the final test observations was used to search for symbolic transformations.

The resulting OOS inflation trajectories are shown below.

![OOF inflation predictions](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/CPI.png)

---

## Step 4. Model stacking using a Markowitz-style approach

A large number of models were used because it is difficult to know in advance which model class will work best on a relatively small macroeconomic dataset.

The candidate set included Ridge, Lasso, Elastic Net, Huber Regression, RANSAC, Bayesian Ridge, ARD Regression, GLSAR, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, SVR, Kernel Ridge, KNN Regressor, Gaussian Process Regression, and several different feature specifications.

Together, this produced **22 candidate combinations for the policy-rate model**. Each policy-rate specification could then be combined with different inflation-model specifications, giving up to

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?22\times22=484" />
</p>

policy-rate/inflation-model combinations for each forecast horizon.

For boosting models and other flexible estimators, I intentionally did not perform a large hyperparameter search. Tree depth was also kept very small. With quarterly macroeconomic data, the number of observations is limited, and aggressive tuning of flexible models could easily lead to overfitting.

### Markowitz-style stacking

Instead of selecting one model and discarding the others, I combine their predictions.

The idea is similar to the **Markowitz portfolio optimization problem**. In portfolio theory, different assets are combined while taking their covariance structure into account. Here, assets are replaced by forecasting models and asset returns are replaced by forecast errors.

For horizon \(h\), let the matrix of OOF predictions be

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?P_h%5Cin%5Cmathbb%7BR%7D%5E%7BT_h%5Ctimes%20M_h%7D" />
</p>

where \(T_h\) is the number of observations and \(M_h\) is the number of candidate models.

The corresponding forecast-error matrix is

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?E_h=P_h-y_h\mathbf{1}^{\top}" />
</p>

and the average error of every model is

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mu_h=\frac{1}{T_h}\sum_{t=1}^{T_h}e_{t,h}" />
</p>

Minimizing only the covariance of the forecast errors would not be enough. A model could have a very stable error but still be systematically biased. For this reason, the optimization uses the second moment of the errors:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?A_h=\Sigma_h^{LW}%2B\mu_h\mu_h^{\top}" />
</p>

where <p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5CSigma_h%5E%7BLW%7D" />
</p> is the covariance matrix estimated using **Ledoit-Wolf shrinkage**.

This follows directly from the MSE decomposition. For a weighted ensemble,

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmathrm%7BMSE%7D(w)%3Dw%5E%7B%5Ctop%7D%5CSigma_h%20w%2B(w%5E%7B%5Ctop%7D%5Cmu_h)%5E2" />
</p>

or equivalently,

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathrm{MSE}(w)=w^{\top}\left(\Sigma_h%2B\mu_h\mu_h^{\top}\right)w" />
</p>

Thus, the objective takes both forecast variance and forecast bias into account.

### Ledoit-Wolf shrinkage and ridge regularization

Forecasts from different models can be strongly correlated. Ridge, Elastic Net, Bayesian Ridge, boosting models, and other estimators may sometimes generate very similar trajectories. With a relatively small number of quarterly observations, this can make the ordinary sample covariance matrix unstable or close to singular.

Ledoit-Wolf shrinkage makes this estimate more stable:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5CSigma_h%5E%7BLW%7D%3D(1-%5Cdelta)S_h%2B%5Cdelta%20F_h" />
</p>

where \(S_h\) is the sample covariance matrix, \(F_h\) is the shrinkage target, and \(\delta\) controls the shrinkage intensity.

The second-moment matrix is then normalized using

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?s_h=\frac{\mathrm{tr}(A_h)}{M_h}" />
</p>

and an additional ridge penalty is added:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?Q_h%3D%5Cfrac%7BA_h%7D%7Bs_h%7D%2B%5Clambda%20I" />
</p>

The ridge term is useful because even after shrinkage many forecasts can remain very similar. It prevents the optimizer from placing very different weights on models only because of small and noisy differences in their estimated errors.

The final weights are obtained from

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?w_h^{*}=\arg\min_{w}\;w^{\top}Q_hw" />
</p>

subject to

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmathbf%7B1%7D%5E%7B%5Ctop%7Dw%3D1%2C%5Cqquad%200%5Cleq%20w_j%5Cleq%201" />
</p>

so all weights are non-negative and sum to one.

The final ensemble forecast is

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{y}_{t,h}^{ensemble}=\sum_{j=1}^{M_h}w_{j,h}^{*}\hat{y}_{t,h}^{(j)}" />
</p>

In this sense, the procedure constructs a portfolio of forecasting models. A model is useful not only when it has a low individual error, but also when its errors contain information that is different from the errors of other models.

The optimization is performed independently for every horizon from \(h=1\) to \(h=8\), so the composition of the model portfolio is allowed to change with the forecast horizon.

---

## Forecast results

The resulting predicted and realized inflation trajectories are shown below.

![Actual and forecast inflation trajectories](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/frcsts.png)

Another useful way to inspect the forecasts is the **hairy plot**. Each line represents a forecast trajectory generated at a different point in time. The figure below shows inflation forecasts up to four quarters ahead.

![Four-quarter-ahead inflation hairy plot](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/hairy_plot.png)

This plot gives more information than a single final forecast line because it shows how the predicted path changed as new observations became available. It also makes forecast instability much easier to notice.

## Comparison with simple benchmarks

As a forecasting model, the proposed approach does not outperform simple time-series benchmarks. AR and ARIMA models achieve lower inflation forecast MSE on the test sample.

| Method | MSE | Observations |
| --- | ---: | ---: |
| AR(4) | 0.000016 | 44 |
| ARIMA(1,0,1) | 0.000016 | 44 |
| Train mean | 0.000016 | 44 |
| AR(1) | 0.000016 | 44 |
| Naive | 0.000019 | 44 |
| **Markowitz stacking** | **0.000031** | **44** |

The proposed model is worse at forecasting inflation than simple benchmarks such as AR and ARIMA. However, these benchmarks only predict inflation and cannot be directly used to construct a **policy rule**. Our pipeline provides a conditional, model-implied inflation response to alternative rate inputs. This allows us to perform scenario analysis within the fitted model, but it does not identify the causal effect of monetary policy.

---

## Step 5. From forecasting to a monetary policy decision

The final step converts the forecasting system into a decision rule.

The central bank does not only want to predict inflation. The actual problem is to choose a policy rate that makes future inflation as close as possible to the target.

Let \(\pi^{target}\) denote the inflation target. For each candidate policy rate \(i\), the model produces the corresponding inflation forecast. The policy problem can then be written as

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?i_t^{*}=\arg\min_i\left(\pi^{target}-\hat{f}_h(X_t,\hat{u}_t^{i}(i))\right)^2" />
</p>

Instead of asking only *what inflation will be*, the model therefore searches over possible policy-rate values and asks which rate produces an inflation trajectory closest to the desired target.

### Oracle validation

As an additional diagnostic, I also considered an oracle version of the problem. Here, realized future inflation is temporarily treated as known, and the model searches over the available rate grid for the rate that gives the closest inflation forecast:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?i_t%5E%7Boracle%7D%3D%5Carg%5Cmin_i%5Cleft(%5Cpi_%7Bt%2Bh%7D%5E%7Brealized%7D-%5Chat%7Bf%7D_h(X_t%2C%5Chat%7Bu%7D_t%5E%7Bi%7D(i))%5Cright)%5E2" />
</p>

The reported oracle MSE is the **inflation-forecast MSE after ex-post rate selection**:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmathrm%7BMSE%7D_%7Boracle%7D%5E%7B%5Cpi%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bt%2Ch%7D%5Cleft%5B%5Cpi_%7Bt%2Bh%7D%5E%7Brealized%7D-%5Chat%7B%5Cpi%7D_%7Bt%2Bh%7D(i_%7Bt%2Ch%7D%5E%7Boracle%7D)%5Cright%5D%5E2" />
</p>

The resulting value is

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmathrm%7BMSE%7D_%7Boracle%7D%5E%7B%5Cpi%7D%3D2.2816596457603746%5Ctimes10%5E%7B-5%7D" />
</p>

This value is measured in squared quarterly `Δlog(CPI)`, not in squared percentage points of the policy rate.

The reported value is **not an MSE between the oracle and observed policy rates**. It is a model-implied lower bound obtained by selecting the best rate ex post over the restricted rate grid. Since future inflation is used to select the rate, it cannot be interpreted as real policy performance.

### Policy-rule comparison

The different rate scenarios can also be compared using their model-implied inflation MSE.

| Policy | Model-implied inflation MSE | Observations |
| --- | ---: | ---: |
| **Ex-post oracle** | **0.000023** | 44 |
| Hold previous-quarter rate | 0.000031 | 44 |
| Observed current rate | 0.000031 | 44 |

The **ex-post oracle** has the lowest error because it selects the rate after future inflation is already known. Therefore, its MSE should be treated as a model-implied lower bound rather than as evidence that the oracle policy could be implemented in practice.

![Policy rule MSE comparison](https://raw.githubusercontent.com/stepan-sapunkov/financial-ml-notes/main/drafts/MP/policy_MSE.png)


---

## Leakage control

Avoiding data leakage is especially important here because information passes through several stages before the final forecast is produced.

The final holdout sample was separated from the model-development sample from the beginning. The reduced dataset available in [train.csv](https://github.com/stepan-sapunkov/financial-ml-notes/blob/main/drafts/MP/train.csv) was used for model development, while the remaining observations were reserved for the final test.

PySR expressions and other features were generated only from the training data. The policy-rate and inflation models were also estimated without access to the final holdout. More importantly, when predictions from one stage were used as inputs to another stage, I used **OOF predictions rather than ordinary in-sample fitted values**. Stacking weights and the Ledoit-Wolf covariance matrix were also estimated from OOF forecast errors rather than final test errors.

This is particularly important for the two-stage structure. If fitted values from the policy-rate model were passed directly into the inflation model, the second stage would receive unrealistically accurate information about the first stage. OOF predictions reproduce the out-of-sample setting much more closely.

The same principle was applied to horizon-specific model and feature selection. The final test targets were not used to choose models, transformations, or stacking weights, and all predictors were based only on information available at the corresponding forecasting date.

A simplified view of the project structure is shown below:

```text
drafts/MP/
│
├── US_dataQ.xlsx
│   └── Full quarterly macroeconomic dataset
│
├── train.csv
│   └── Reduced dataset used for model development
│
├── EDA_rate.ipynb
│   └── EDA and feature generation for the policy-rate model
│
├── rate_modls.ipynb
│   └── Policy-rate models and OOF predictions
│       └── rate.png
│
├── EDA_infl.ipynb
│   └── EDA and feature generation for the inflation model
│
├── infl_modls.ipynb
│   └── Inflation models and OOF predictions
│       └── CPI.png
│
├── stacking.ipynb
│   └── Markowitz-style model stacking
│
├── decision_layer.ipynb
│   └── Policy-rate optimization and final decision layer
│
├── data.png
│   └── Original macroeconomic time series
│
├── scatter.png
│   └── Example of a PySR-generated feature
│
├── frcsts.png
│   └── Final actual vs predicted inflation trajectories
│
└── hairy_plot.png
    └── Rolling four-quarter-ahead inflation forecasts
```

## Final pipeline

The complete procedure can be summarized more compactly as

```text
Macroeconomic data
        → Stationary transformations and lags
        → PySR feature generation
        → Policy-rate model
        → Inflation models
        → Horizon-specific Markowitz stacking
        → Inflation response under alternative rates
        → Policy-rate optimization
```
The policy-rate stage produces the unexpected policy component

<p align="center"> <img src="https://latex.codecogs.com/svg.image?%5Chat%7Bu%7D_t%5E%7Bi%7D%3Di_t-%5Chat%7Bg%7D(X_t)" /> </p>

which is then used inside the inflation model

<p align="center"> <img src="https://latex.codecogs.com/svg.image?%5Chat%7B%5Cpi%7D_%7Bt%2Bh%7D%3Df_h%5Cleft(X_t%2C%5Chat%7Bu%7D_t%5E%7Bi%7D%5Cright)" /> </p>

The final decision layer searches over possible policy-rate values and chooses the rate that makes predicted inflation closest to the target:

<p align="center"> <img src="https://latex.codecogs.com/svg.image?i_t%5E%7B*%7D%3D%5Carg%5Cmin_i%5Cleft(%5Cpi%5E%7Btarget%7D-%5Chat%7B%5Cpi%7D_%7Bt%2Bh%7D(i)%5Cright)%5E2" /> </p>
The main idea is to treat monetary policy not as one isolated forecasting problem, but as a connected system. First, the model estimates the predictable part of the policy rate and extracts its unexpected component. This information is then used together with macroeconomic variables to model future inflation. Instead of relying on one estimator, many candidate specifications are combined using covariance-aware stacking. Finally, the resulting inflation model is used as a response surface over which the policy rate can be optimized.

The approach is intentionally different from a structural macroeconomic model. It does not replace the identification assumptions of SVAR or DSGE models. Instead, the experiment studies how far a carefully designed and leakage-controlled ML pipeline can go in learning a useful monetary-policy decision rule from historical data.


## Robustness and limitations

The model is trained on **73 quarterly observations from 2006Q1 to 2024Q1**. The final holdout contains **9 quarters from 2024Q2 to 2026Q2**.

Across all forecast horizons, the test produces **44 origin–horizon forecast pairs**. Their distribution is `h1 = 9`, `h2 = 8`, `h3 = 7`, `h4 = 6`, `h5 = 5`, `h6 = 4`, `h7 = 3`, and `h8 = 2`. These 44 forecast errors should not be treated as 44 independent observations because forecasts from different horizons overlap and share the same forecast origins. Given the small holdout sample, I therefore do not make claims about statistical significance.

The stacking procedure also needs additional robustness checks. In particular, it is useful to compare the results across different values of the ridge parameter `lambda`, against simple equal-weight stacking, and after removing individual model families. This can show whether the final result depends strongly on a particular regularization choice or group of models.

The dimensionality of the stacking problem is another limitation. For the inflation model there can be up to **484 candidate combinations**, while only around **51–65 training observations** are available depending on the horizon. Therefore, the stacking problem is strongly high-dimensional, with `M ≫ T`. Ledoit–Wolf shrinkage and ridge regularization make the optimization numerically more stable, but they do not remove the risk of overfitting.

Another important limitation is the macroeconomic data itself. The current experiment uses **revised historical data**, rather than the information that was actually available to policymakers at each point in time. A realistic monetary-policy backtest would require real-time vintages, for example from **ALFRED**, and should account for publication delays and later revisions of variables such as GDP and CPI.

Finally, the information set must be defined carefully. The current setup effectively assumes that the variables assigned to quarter `t` are available when the forecast is made. In a real-time experiment this assumption may be too strong, especially for GDP and other variables released with a delay. A stricter backtest should construct the feature set using only information that had actually been published at each forecast origin.
