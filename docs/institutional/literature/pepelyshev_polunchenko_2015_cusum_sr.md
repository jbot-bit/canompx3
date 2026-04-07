# Real-time Financial Surveillance via Quickest Change-Point Detection — Pepelyshev & Polunchenko (2015)

**Source:** `resources/real_time_strategy_monitoring_cusum.pdf`
**Authors:** Andrey Pepelyshev (Cardiff University), Aleksey S. Polunchenko (SUNY Binghamton)
**Publication:** *Statistics and Its Interface*, Vol. 0 (2015), pp. 1–14; arXiv:1509.01570
**Extracted:** 2026-04-07
**Pages read:** 1-14 (full paper)

**Criticality:** 🟡 **MEDIUM-HIGH** — provides the Shiryaev-Roberts (SR) procedure for detecting regime drift in live strategies. Needed for provisional deployment monitoring.

---

## Abstract (verbatim, page 1)

> "We consider the problem of efficient financial surveillance aimed at 'on-the-go' detection of structural breaks (anomalies) in 'live'-monitored financial time series. With the problem approached statistically, viz. as that of *multi-cyclic sequential (quickest) change-point detection*, we propose a semi-parametric multi-cyclic change-point detection procedure to promptly spot anomalies as they occur in the time series under surveillance. The proposed procedure is a derivative of the likelihood ratio-based Shiryaev–Roberts (SR) procedure; the latter is a quasi-Bayesian surveillance method known to deliver the fastest (in the multi-cyclic sense) speed of detection, whatever be the false alarm frequency."
>
> **Keywords:** CUSUM chart, Financial surveillance, Sequential analysis, Shiryaev–Roberts procedure, Quickest change-point detection.

---

## CUSUM statistic (page 3 verbatim)

> "More specifically, the CUSUM chart is based on the statistic W_n ≜ max{0, log V_n}, where V_n is as in (1). Note that W_n satisfies the recurrence
>
> **W_n ≜ max{0, W_{n-1} + ℒ_n}, n ≥ 1, W_0 = 0.**   (Equation 3)
>
> The corresponding stopping rule is
>
> **𝒞_h ≜ min{n ≥ 1: W_n ≥ h},**   (Equation 4)
>
> where h > 0 is a detection threshold preset so as to achieve the desired level γ > 1 of false alarm, and thus guarantee 𝒞_h ∈ Δ(γ)."

## Shiryaev-Roberts (SR) statistic (page 3 verbatim)

> "The SR rule calls for stopping at epoch
>
> **𝒮_A ≜ min{n ≥ 1: R_n ≥ A},**   (Equation 10)
>
> where the SR statistic {R_n}_{n≥0} is given by the recursion
>
> **R_n ≜ (1 + R_{n-1}) Λ_n, n ≥ 1, R_0 = 0;**   (Equation 11)
>
> cf. [42, 43] and [39]; here A > 0 is a detection threshold set a priori so as to ensure 𝒮_A ∈ Δ(γ)."

## Why SR is better than CUSUM (page 3)

> "As has been shown in [30, 31, 45], the Shiryaev–Roberts (SR) procedure [42, 43, 39] is *exactly* optimal for every γ > 1 with respect to the stationary average detection delay STADD(T). Thus, in the multi-cyclic setting the SR procedure is a better alternative to the popular CUSUM chart."

---

## Nonparametric score function approach (pages 5-6 verbatim)

> "The above suggests that when it is impossible to construct a LR, the latter can be replaced with a computable score function S_n ≜ S_n(X_1, ..., X_n) such that 𝔼_∞[S_n] < 0 for all n ≥ 1 and 𝔼_ν[S_n] > 0 for all 0 ≤ ν < n with n ≥ 1. This is the key element of the nonparametric approach, and in the context of quickest change-point detection this idea has been previously suggested and explored, e.g., by McDonald [22], Lai [18], Gordon and Pollak [12, 13], and recently also by Pollak [28]."

### Score function modified SR procedure (Equation 13-14)

> "More generally, for any appropriately designed score function S_n, the original SR statistic {R_n}_{n≥0} given by (11) can be replaced with
>
> **R̃_n ≜ (1 + R̃_{n-1}) e^{S_n}, n ≥ 1, R̃_0 = 0,**   (Equation 13)
>
> so that the corresponding SR stopping time is the form
>
> **𝒮̃_A ≜ min{n ≥ 1: R̃_n ≥ A},**   (Equation 14)
>
> where A > 0 is again the detection threshold."

### Score function modified CUSUM (Equation 15-16)

> "Likewise, for the CUSUM chart, the original CUSUM statistic {W_n}_{n≥0} given by (3) can be replaced with
>
> **W̃_n ≜ max{0, W̃_{n-1} + S_n}, n ≥ 1, W̃_0 = 0,**   (Equation 15)
>
> so that the corresponding CUSUM stopping time becomes
>
> **𝒞̃_h ≜ min{n ≥ 1: W̃_n ≥ h},**   (Equation 16)
>
> where h > 0 is again the detection threshold."

### Linear-quadratic score for mean+variance change (page 6)

> "To illustrate this point, suppose we are interested in detecting a change in both the mean and variance of the observations. Let μ_∞ ≜ 𝔼_∞[X_n] and σ²_∞ ≜ Var_∞[X_n], and μ ≜ 𝔼_0[X_n] and σ² ≜ Var_0[X_n] denote the pre- and post-anomaly mean values and variances, respectively. Introduce X̄_n ≜ (X_n - μ_∞)/σ_∞, i.e., the centered and standardized n-th data point. To deal with the uncertainty in μ and σ², consider the following linear-quadratic score function
>
> **S_n(X̄_n) = C_1·X̄_n + C_2·X̄_n² - C_3,**   (Equation 17)
>
> where C_1, C_2 and C_3 are design parameters; cf. [52]."

### Optimal parameters (Equation 18)

> "Note that the score function S_n given by (17) with
>
> **C_1 = δq², C_2 = (1-q²)/2, C_3 = (δ²q²/2 - log q),**   (Equation 18)
>
> where q ≜ σ_∞/σ, δ ≜ (μ - μ_∞)/σ_∞, is optimal if the pre- and post-change distributions are Gaussian with known μ and σ². This is true because the score function S_n given by (17) is then simply nothing but the LLR. [...] In the case when the variance either does not change at all or changes relatively insignificantly compared to the magnitude of the change in the mean, the coefficient C_2 may be set equal to zero. This appears to be typical for many applications."

---

## Case study: HST stock 2003 structural break (page 10-11)

> "Via a simple Monte Carlo experiment we estimated that setting A ≈ 60 and h ≈ 0.3 ensures that the ARL to false alarm of either procedure is approximately 7 samples, which is roughly a week, since the timescale is working days."

### Detection delay comparison (page 10 quote)

> "The detection process is illustrated in Figure 10. Specifically, Figure 10a shows the behavior of the SR statistic in a short time window covering March 13, 2003, i.e., the date at which the HST stock underwent the change we would like to detect. [...] Figure 10b shows the same but for the CUSUM statistic. We see that both procedures successfully detect the onset of the drift (occurring on March 13, 2003), and the detection delays are about one day each."

---

## Conclusion (page 12 verbatim)

> "We considered the problem of rapid but reliable anomaly detection in 'live' financial data. We treated the problem statistically, viz. as that of quickest change-point detection, and proposed an anomaly-detection method that derives from the multi-cyclic (repeated) Shiryaev–Roberts (SR) detection procedure. We decided to go with this largely neglected near-coeval of the celebrated CUSUM and EWMA charts because of the strong multi-cyclic optimality properties that the SR procedure was recently discovered to have under the basic iid change-point detection setup; no such properties are exhibited by either the 'good old' CUSUM 'inspection scheme' or the EWMA chart. To handle real-world financial data, the proposed SR-derivative utilizes the information contained in the data in the SR-like Bayesian manner with the likelihood ratio replaced with a change-sensitive score function."

---

## Application to our project (synthesis)

### For live strategy drift monitoring

Use the SR procedure (Equation 11) on each deployed strategy's rolling R-multiple returns. Design:
- **Pre-anomaly μ_∞, σ_∞:** computed from first 100 live trades per strategy
- **Post-anomaly μ, σ:** a threshold shift that constitutes concerning regime change (e.g., mean drops by 1σ_∞)
- **Score function:** Equation 17 with coefficients per Equation 18
- **Threshold A:** calibrated via Monte Carlo so ARL to false alarm = ~60 trading days (one false alarm per ~3 months on average)
- **Detection delay goal:** ~5-10 trades after actual drift (tuneable via A)

### For portfolio-level regime detection

SR on portfolio equity curve, same design. If fires: pause all copies pending investigation.

### Integration point

A new script `trading_app/live/cusum_monitor.py` implementing the SR procedure, reading recent trades from `paper_trades` or live broker, and writing alerts. Runs as a daily scheduled job.

---

## Related literature
- `lopez_de_prado_2020_ml_for_asset_managers.md` — discusses change-point detection in the ML context
- `bailey_et_al_2013_pseudo_mathematics.md` — establishes the need for forward monitoring given backtest weaknesses
