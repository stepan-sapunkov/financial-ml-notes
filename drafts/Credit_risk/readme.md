# Credit Risk Estimation with Structural Models

Suppose we want to estimate the credit risk of a company. Such an estimate can be useful for CDS pricing, early detection of deterioration in the borrower’s financial condition, and refinement of credit ratings.

One way to estimate credit risk from market information is to use structural credit risk models, such as the Merton model and its extensions, including [CreditGrades](https://www.msci.com/documents/10199/dd31bcce-6fe3-47b7-9fb7-10c4c8f750ba).

In both models, the value of the company’s assets is assumed to follow a geometric Brownian motion. Under the physical probability measure, it can be written as

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{dV_t}{V_t}=\mu\,dt+\sigma_V\,dW_t^P" />
</p>

This means that log returns are normally distributed. Therefore, if we estimate the parameters `mu` and `sigma_V` sufficiently well, we can describe the physical distribution of future asset values.

One simple way to see this is to simulate many possible paths of the asset value and then look at the distribution of terminal values. In Figure 1, I set the initial asset value to 100 and the default barrier to 85. By simulating many paths, we obtain the distribution of the asset value at the end of the horizon and can estimate the physical probability that the asset value falls below the default barrier.

![Simulated asset value distribution](1.png)

However, this approach has several strong limitations.

First, the dynamics of the company may experience structural breaks. In this case, parameters estimated from historical data may no longer describe the current process well. One possible extension is to introduce jumps into the dynamics.

Second, the parameters themselves may change over time and depend on the current state of the company or the market. Instead of assuming constant parameters, one can write

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mu_t=f_\mu(X_t),\qquad\sigma_{V,t}=f_\sigma(X_t)" />
</p>

where `X_t` is a set of observable features. For example, `f_mu` and `f_sigma` can be estimated using boosting or a neural network trained to predict drift and volatility from current information.

For pricing credit instruments, however, structural models are usually considered under the risk-neutral measure. In this case, the asset dynamics can be written as

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\frac{dV_t}{V_t}=(r_t-\delta_t)\,dt+\sigma_V\,dW_t^Q" />
</p>

where `r_t` is the risk-free rate and `delta_t` represents payouts from the firm. Thus, the physical drift `mu` is relevant for forecasting physical default probabilities, but does not directly enter the risk-neutral pricing formulas used below.

In the Merton model, the basic idea is very simple. Let `V_T` be the value of the company’s assets at the debt maturity date and `D` be the amount of debt that must be repaid. Default occurs when

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?V_T%3CD" />
</p>

while the equity value at maturity is

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?E_T=\max(V_T-D,0)" />
</p>

Economically, this is natural. If the company owns assets worth more than its debt, it can repay creditors and the remaining value belongs to shareholders. If the value of the assets is below the amount of debt, the company cannot fully repay its obligations. Shareholders have limited liability, so their payoff is zero, while creditors bear the loss.

Therefore, equity can be interpreted as a call option on the company’s assets with strike equal to the debt level.

Figure 2 illustrates the sensitivity of the Merton-implied credit spread to asset volatility and to different hypothetical paths of the firm’s asset value. The drift parameter `mu` is used only to generate these illustrative asset-value paths; it does not directly enter the risk-neutral Merton pricing formula.

![Merton spread sensitivity](2.png)

In my case, however, I do not focus on improving the Merton model itself. My goal was to test whether the CreditGrades approach works reasonably well in practice.

An important extension of CreditGrades relative to the basic Merton framework is that the default barrier is itself uncertain rather than fixed. Thus, the model introduces an additional source of uncertainty into the default mechanism.

For the empirical example, I use Boeing. It seems to be a useful case because the company experienced periods of significant credit deterioration, while at the same time being large enough for both its stock and CDS contracts to have relatively liquid market prices.

![Boeing market data](3.png)

Structural models are generally sensitive to their parameters. Therefore, an important practical question is how these parameters should be estimated.

In CreditGrades, the CDS spread can be written as a function of observable market variables and several structural parameters:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?c_t=c(S_t,D_t,r_t,\sigma_{S,t};\lambda,\bar{L},R,T)" />
</p>

where `S_t` is the stock price, `D_t` is debt per share, `r_t` is the risk-free rate, and `sigma_{S,t}` is equity volatility. The parameters `lambda` and `L` control uncertainty in the default barrier and its average level.

The figure below shows that CreditGrades estimates can change substantially when these assumptions are changed. I vary `lambda`, `L`, recovery rate `R`, and the level of equity volatility.

![CreditGrades sensitivity to parameters](4.png)

This sensitivity means that parameter estimation is an important part of the model.

More advanced approaches could allow structural parameters to change over time or depend on observable information. Another possibility is to recalibrate the model sequentially using a walk-forward procedure. Here I use a simpler experiment. I split the sample into

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{T}_{train}=2001\text{-}2009,\qquad\mathcal{T}_{test}=2010\text{-}2020" />
</p>

and estimate the CreditGrades parameters only once on the training sample.

I keep the recovery rate and CDS maturity fixed at

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?R=0.5%2C%5Cquad%20T=5" />
</p>

and calibrate `lambda` and `L` by minimizing the squared error between log model spreads and log market CDS spreads:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Chat%7B%5Ctheta%7D%3D%5Carg%5Cmin_%7B%5Ctheta%7D%5Cfrac%7B1%7D%7BN_%7Btrain%7D%7D%5Csum_%7Bt%5Cin%20train%7D%5B%5Clog%20c_t%5E%7BCG%7D(%5Ctheta)-%5Clog%20c_t%5E%7Bmkt%7D%5D%5E2" />
</p>

The baseline CreditGrades parameters are

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Clambda=0.30%2C%5Cquad%20%5Cbar%7BL%7D=0.75" />
</p>

while calibration on 2001–2009 gives approximately

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Chat%7B%5Clambda%7D=2.047%2C%5Cquad%20%5Chat%7B%5Cbar%7BL%7D%7D=0.398" />
</p>

These parameters are then fixed for the entire 2010–2020 test period. Thus, the test does not use future CDS information to update the CreditGrades parameters.

The out-of-sample comparison is shown below.

![Out-of-sample CreditGrades comparison](5.png)

The calibrated model follows the market CDS spread considerably more closely than the baseline specification:

| Metric | Baseline CG | Calibrated once |
| --- | ---: | ---: |
| RMSE, bps | 141.879 | 50.593 |
| MAE, bps | 72.765 | 32.788 |
| Bias, bps | -7.831 | -30.536 |

I also test whether the model can be useful for a simple CDS-equity hedge.

On the training sample, I estimate the relation between changes in the model CDS spread and changes in the stock price:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5CDelta%20c_t%5E%7BCG%7D%3D%5Calpha%2B%5Cbeta%5CDelta%20S_t%2B%5Cvarepsilon_t" />
</p>

The estimated `beta` is then fixed during the test period. Since the estimated relation is negative, an increase in the stock price is associated with a decrease in credit spreads.

The CDS position is converted into an equity hedge using the CDS risky annuity:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?h_t=-A_t^{mkt}(T)\,10^{-4}\widehat{\beta}" />
</p>

The resulting out-of-sample hedge performance is:

| Metric | Baseline CG | Calibrated once |
| --- | ---: | ---: |
| Unhedged annual vol / notional | 0.0394 | 0.0394 |
| Hedged annual vol / notional | 0.0380 | 0.0355 |
| Variance reduction | 0.0704 | 0.1861 |

Therefore, even a simple one-time calibration substantially improves the out-of-sample fit of CreditGrades to observed CDS spreads. It also improves the effectiveness of the CDS-equity hedge: variance reduction increases from about 7% for the baseline model to about 19% for the calibrated model.

At the same time, the experiment shows an important limitation of structural models: their practical performance can depend strongly on structural parameter assumptions. This makes parameter estimation and parameter stability important questions for further improvement.
