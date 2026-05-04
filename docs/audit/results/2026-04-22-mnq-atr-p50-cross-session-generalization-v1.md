# MNQ ATR_P50 Cross-Session Generalization Audit

**Date:** 2026-04-22
**Prereg lock:** `docs/audit/hypotheses/2026-04-22-mnq-atr-p50-cross-session-generalization-v1.yaml` (`commit_sha=1be7e6db`)

## Verdict

`LANE_LOCAL_ONLY`

This is a fixed-family audit of ATR_P50 as an `R1` overlay across all
MNQ sessions on the two apertures the repo already uses for this class:
`O15` and `O30`.

## Aperture O15

- IS rows: `15551`
- Powered-session median `delta_expr`: `+0.0252`
- Aperture verdict: `LANE_LOCAL_ONLY`

| session        | n_total_is | n_fire | n_nonfire | expR_fire | expR_nonfire | delta_expr | wr_fire | wr_nonfire | welch_p_raw | welch_q_local | powered |
| -------------- | ---------- | ------ | --------- | --------- | ------------ | ---------- | ------- | ---------- | ----------- | ------------- | ------- |
| BRISBANE_1025  | 1721       | 916    | 805       | -0.0057   | -0.0329      | 0.0272     | 0.4378  | 0.4472     | 0.6099      | 0.8659        | True    |
| CME_PRECLOSE   | 201        | 111    | 90        | 0.1683    | -0.0339      | 0.2021     | 0.4865  | 0.4111     | 0.2299      | 0.8659        | True    |
| CME_REOPEN     | 506        | 311    | 195       | 0.0144    | -0.0578      | 0.0721     | 0.4309  | 0.4205     | 0.4867      | 0.8659        | True    |
| COMEX_SETTLE   | 1498       | 796    | 702       | -0.0030   | 0.0281       | -0.0312    | 0.4209  | 0.4430     | 0.6048      | 0.8659        | True    |
| EUROPE_FLOW    | 1718       | 912    | 806       | 0.0491    | 0.0157       | 0.0334     | 0.4507  | 0.4467     | 0.5474      | 0.8659        | True    |
| LONDON_METALS  | 1709       | 908    | 801       | 0.0530    | 0.0298       | 0.0233     | 0.4460  | 0.4457     | 0.6798      | 0.8659        | True    |
| NYSE_CLOSE     | 231        | 112    | 119       | -0.0134   | -0.0325      | 0.0190     | 0.4196  | 0.4202     | 0.9005      | 0.9776        | True    |
| NYSE_OPEN      | 1393       | 767    | 626       | 0.0297    | 0.1147       | -0.0850    | 0.4211  | 0.4601     | 0.1919      | 0.8659        | True    |
| SINGAPORE_OPEN | 1719       | 913    | 806       | 0.1130    | -0.0207      | 0.1337     | 0.4841  | 0.4491     | 0.0136      | 0.1628        | True    |
| TOKYO_OPEN     | 1721       | 915    | 806       | 0.0767    | 0.0313       | 0.0455     | 0.4634  | 0.4603     | 0.4092      | 0.8659        | True    |
| US_DATA_1000   | 1495       | 805    | 690       | 0.1055    | 0.1073       | -0.0017    | 0.4559  | 0.4638     | 0.9776      | 0.9776        | True    |
| US_DATA_830    | 1639       | 883    | 756       | 0.0053    | -0.0151      | 0.0204     | 0.4247  | 0.4259     | 0.7216      | 0.8659        | True    |

### OOS descriptive only

| session        | n_total_oos | n_fire_oos | n_nonfire_oos | expR_fire_oos | expR_nonfire_oos | delta_expr_oos |
| -------------- | ----------- | ---------- | ------------- | ------------- | ---------------- | -------------- |
| BRISBANE_1025  | 72          | 59         | 13            | 0.2947        | 0.0732           | 0.2215         |
| CME_PRECLOSE   | 13          | 12         | 1             | 0.4123        | -1.0000          | 1.4123         |
| CME_REOPEN     | 38          | 27         | 11            | 0.2475        | 0.5100           | -0.2625        |
| COMEX_SETTLE   | 60          | 48         | 12            | -0.0910       | -0.1936          | 0.1026         |
| EUROPE_FLOW    | 72          | 59         | 13            | 0.2981        | -0.0885          | 0.3866         |
| LONDON_METALS  | 72          | 59         | 13            | -0.0129       | 0.1224           | -0.1353        |
| NYSE_CLOSE     | 11          | 8          | 3             | 0.1951        | 0.5813           | -0.3862        |
| NYSE_OPEN      | 48          | 40         | 8             | 0.1737        | 0.5356           | -0.3619        |
| SINGAPORE_OPEN | 72          | 59         | 13            | 0.0492        | 0.2518           | -0.2026        |
| TOKYO_OPEN     | 72          | 59         | 13            | 0.0603        | 0.0997           | -0.0394        |
| US_DATA_1000   | 60          | 48         | 12            | 0.2290        | 0.0178           | 0.2112         |
| US_DATA_830    | 70          | 57         | 13            | -0.2827       | 0.2893           | -0.5720        |

## Aperture O30

- IS rows: `14058`
- Powered-session median `delta_expr`: `NaN`
- Aperture verdict: `SGP_FAILS_RAW`

| session        | n_total_is | n_fire | n_nonfire | expR_fire | expR_nonfire | delta_expr | wr_fire | wr_nonfire | welch_p_raw | welch_q_local | powered |
| -------------- | ---------- | ------ | --------- | --------- | ------------ | ---------- | ------- | ---------- | ----------- | ------------- | ------- |
| BRISBANE_1025  | 1715       | 913    | 802       | 0.0167    | -0.0388      | 0.0554     | 0.4348  | 0.4277     | 0.3138      | 0.9415        | True    |
| CME_PRECLOSE   | 40         | 22     | 18        | -0.1188   | -0.0663      | -0.0525    | 0.3636  | 0.3889     | 0.8912      | 0.9486        | False   |
| CME_REOPEN     | 333        | 216    | 117       | -0.0861   | -0.1042      | 0.0182     | 0.3843  | 0.3932     | 0.8892      | 0.9486        | True    |
| COMEX_SETTLE   | 1222       | 654    | 568       | -0.0711   | -0.0017      | -0.0693    | 0.3869  | 0.4243     | 0.3009      | 0.9415        | True    |
| EUROPE_FLOW    | 1699       | 904    | 795       | 0.0063    | 0.0312       | -0.0249    | 0.4248  | 0.4453     | 0.6603      | 0.9486        | True    |
| LONDON_METALS  | 1677       | 890    | 787       | 0.0595    | 0.0185       | 0.0410     | 0.4427  | 0.4333     | 0.4773      | 0.9486        | True    |
| NYSE_CLOSE     | 131        | 70     | 61        | -0.1362   | 0.0129       | -0.1491    | 0.3714  | 0.4426     | 0.4568      | 0.9486        | True    |
| NYSE_OPEN      | 1037       | 580    | 457       | 0.0105    | 0.0337       | -0.0232    | 0.4121  | 0.4245     | 0.7583      | 0.9486        | True    |
| SINGAPORE_OPEN | 1707       | 905    | 802       | 0.1243    | 0.0164       | 0.1079     | 0.4785  | 0.4489     | 0.0534      | 0.6411        | True    |
| TOKYO_OPEN     | 1705       | 908    | 797       | 0.0663    | 0.0007       | 0.0656     | 0.4515  | 0.4366     | 0.2427      | 0.9415        | True    |
| US_DATA_1000   | 1181       | 636    | 545       | 0.0372    | 0.0574       | -0.0202    | 0.4245  | 0.4385     | 0.7734      | 0.9486        | True    |
| US_DATA_830    | 1611       | 868    | 743       | -0.0235   | -0.0197      | -0.0038    | 0.4078  | 0.4172     | 0.9486      | 0.9486        | True    |

### OOS descriptive only

| session        | n_total_oos | n_fire_oos | n_nonfire_oos | expR_fire_oos | expR_nonfire_oos | delta_expr_oos |
| -------------- | ----------- | ---------- | ------------- | ------------- | ---------------- | -------------- |
| BRISBANE_1025  | 72          | 59         | 13            | 0.1065        | 0.6331           | -0.5267        |
| CME_PRECLOSE   | 3           | 3          | 0             | 1.4294        | NaN              | NaN            |
| CME_REOPEN     | 29          | 21         | 8             | 0.0377        | 0.4958           | -0.4581        |
| COMEX_SETTLE   | 45          | 38         | 7             | 0.0905        | 0.3873           | -0.2967        |
| EUROPE_FLOW    | 72          | 59         | 13            | 0.1094        | 0.1064           | 0.0030         |
| LONDON_METALS  | 72          | 59         | 13            | 0.1590        | -0.0608          | 0.2199         |
| NYSE_CLOSE     | 3           | 2          | 1             | 1.4244        | 1.3357           | 0.0886         |
| NYSE_OPEN      | 29          | 26         | 3             | 0.0472        | 0.6487           | -0.6015        |
| SINGAPORE_OPEN | 72          | 59         | 13            | 0.0242        | 0.2873           | -0.2631        |
| TOKYO_OPEN     | 72          | 59         | 13            | 0.2738        | 0.2902           | -0.0163        |
| US_DATA_1000   | 38          | 32         | 6             | 0.0012        | -0.5946          | 0.5958         |
| US_DATA_830    | 70          | 57         | 13            | -0.2746       | 0.6818           | -0.9563        |

## Decision

- Overall family verdict: `LANE_LOCAL_ONLY`
- O15 verdict: `LANE_LOCAL_ONLY`
- O30 verdict: `SGP_FAILS_RAW`
- OOS is reported descriptively only and does not vote in the verdict.
- No threshold search, aperture widening beyond O15/O30, or role drift was performed.
