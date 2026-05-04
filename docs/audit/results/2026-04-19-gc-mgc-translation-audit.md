# GC -> MGC Translation Audit

Date: 2026-04-19

## Scope

Audit the unresolved gold translation question from canonical overlap-era truth:

- does `GC` stay strong in the actual `MGC` overlap era, or was the old proxy edge mostly old history?
- do price-safe triggers transfer from `GC` to `MGC` cleanly?
- if translation fails, is the break at trigger timing, payoff shape, or modeled friction?

This is a research audit only. It is not a discovery run, not a deployment memo, and not a new justification for widening proxy use.

## Guardrails

- Canonical proof uses only `gold.db::orb_outcomes`, `gold.db::daily_features`, and `pipeline.cost_model`.
- 2026 holdout is excluded from selection and diagnosis here (`trading_day < 2026-01-01`).
- Prior docs and old handoffs were used only as orientation/comparison, not as proof.
- No claim is made about `GC` 15m/30m proxy transfer because the canonical `GC` proxy surface here is 5-minute only.

## Executive Verdict

`GC` strength is still real in the overlap era, so the old gold signal was not just a stale-history artifact. But the `GC -> MGC` bridge does not fail because price-safe filters stop firing. It fails mainly because the 5-minute `MGC` payoff shape is materially worse: win rates are modestly lower, average wins are much smaller, and the broad positive `GC` expectancy compresses toward flat or negative on `MGC`.

On the exact retired `GC` validated rows, only 5/17 keep a positive sign on `MGC` overlap, and only 0 of those are above `RR=1.0`. So the shorthand "edge does not transfer" was too blunt, but the stronger claim still holds: the full `GC` proxy shelf does **not** transfer cleanly to `MGC`, and any surviving bridge is narrow, weak, and concentrated at low RR.

## Source-of-Truth Chain

1. `gold.db::orb_outcomes`
2. `gold.db::daily_features`
3. `pipeline.cost_model`

Orientation only:

4. `validated_setups` for the already-retired `GC` rows
5. `docs/plans/2026-04-10-mgc-proxy-hypothesis-design.md`
6. `docs/handoffs/2026-04-10-gc-proxy-discovery-handover.md`

## Finding 1 — The old GC strength is still real in the overlap era

The overlap-era `GC` baseline remains positive across multiple gold-relevant sessions on the canonical 5-minute `E2 / CB1` surface. So this is not just a pre-2022 history artifact.

```text
symbol      orb_label rr_target   n   avg_r  t_stat win_rate avg_win_r avg_loss_r
    GC     CME_REOPEN    1.0000 821  0.0315  0.9607   0.3581    0.8841    -1.0000
    GC     CME_REOPEN    1.5000 821 -0.0282 -0.6955   0.2375    1.3572    -1.0000
    GC     CME_REOPEN    2.0000 821 -0.1043 -2.2670   0.1717    1.8269    -1.0000
    GC   COMEX_SETTLE    1.0000 914  0.0721  2.3211   0.5481    0.8917    -1.0000
    GC   COMEX_SETTLE    1.5000 914  0.0178  0.4601   0.3993    1.3619    -1.0000
    GC   COMEX_SETTLE    2.0000 914 -0.0094 -0.2098   0.3118    1.8363    -1.0000
    GC    EUROPE_FLOW    1.0000 917  0.1045  3.3769   0.5823    0.8966    -1.0000
    GC    EUROPE_FLOW    1.5000 917  0.0935  2.3919   0.4613    1.3704    -1.0000
    GC    EUROPE_FLOW    2.0000 917  0.1079  2.3517   0.3893    1.8457    -1.0000
    GC  LONDON_METALS    1.0000 917  0.0554  1.7653   0.5529    0.9089    -1.0000
    GC  LONDON_METALS    1.5000 917  0.0435  1.1110   0.4373    1.3862    -1.0000
    GC  LONDON_METALS    2.0000 917  0.0657  1.4361   0.3719    1.8627    -1.0000
    GC      NYSE_OPEN    1.0000 918  0.1096  3.4403   0.5664    0.9482    -1.0000
    GC      NYSE_OPEN    1.5000 918  0.0864  2.1618   0.4379    1.4350    -1.0000
    GC      NYSE_OPEN    2.0000 918  0.0944  2.0226   0.3584    1.9207    -1.0000
    GC SINGAPORE_OPEN    1.0000 918  0.0707  2.2526   0.5599    0.9122    -1.0000
    GC SINGAPORE_OPEN    1.5000 918  0.0444  1.1331   0.4368    1.3909    -1.0000
    GC SINGAPORE_OPEN    2.0000 918  0.0445  0.9759   0.3638    1.8709    -1.0000
    GC     TOKYO_OPEN    1.0000 918  0.0539  1.7505   0.5621    0.8749    -1.0000
    GC     TOKYO_OPEN    1.5000 918  0.0342  0.8881   0.4412    1.3441    -1.0000
    GC     TOKYO_OPEN    2.0000 918  0.0284  0.6339   0.3660    1.8097    -1.0000
    GC   US_DATA_1000    1.0000 917  0.1087  3.4062   0.5420    0.9497    -1.0000
    GC   US_DATA_1000    1.5000 917  0.0938  2.3428   0.4057    1.4345    -1.0000
    GC   US_DATA_1000    2.0000 917  0.0562  1.2121   0.3108    1.9203    -1.0000
    GC    US_DATA_830    1.0000 917  0.0455  1.4176   0.5049    0.9464    -1.0000
    GC    US_DATA_830    1.5000 917  0.0719  1.8023   0.4024    1.4314    -1.0000
    GC    US_DATA_830    2.0000 917  0.1070  2.2880   0.3391    1.9152    -1.0000
```

## Finding 2 — Trigger parity is strong; the bridge does not break because filters disappear

Price-safe feature means are nearly identical between `GC` and `MGC` in the overlap era, and the pass-day counts on the retired `GC` filter winners are almost one-for-one.

Modeled friction is also not the culprit: `GC` and `MGC` have the same friction in price points (0.574) and the same minimum-risk floor in price points (1.0). The dollar costs differ by 10x, but the R-multiple burden is the same by construction.

Feature parity:

```text
symbol    n atr20_avg overnight_range_avg prev_day_range_avg nyse_open_orb_avg us_data_1000_orb_avg europe_flow_orb_avg london_metals_orb_avg
    GC 1040   34.6434             18.2778            35.0113            4.3687               4.8576              2.2424                2.4691
   MGC 1039   34.8470             18.0618            34.8937            4.3719               4.9562              2.2377                2.4479
```

Price-safe pass-day parity on retired `GC` winners:

```text
                           strategy_id    orb_label filter_type       GC      MGC delta_days mgc_to_gc_ratio
 GC_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50  EUROPE_FLOW   OVNRNG_50  55.0000  52.0000     3.0000          0.9455
     GC_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50    NYSE_OPEN     ATR_P50 725.0000 702.0000    23.0000          0.9683
     GC_NYSE_OPEN_E2_RR1.5_CB1_ATR_P50    NYSE_OPEN     ATR_P50 725.0000 702.0000    23.0000          0.9683
     GC_NYSE_OPEN_E2_RR2.0_CB1_ATR_P50    NYSE_OPEN     ATR_P50 725.0000 702.0000    23.0000          0.9683
     GC_NYSE_OPEN_E2_RR1.0_CB1_ATR_P70    NYSE_OPEN     ATR_P70 473.0000 480.0000    -7.0000          1.0148
     GC_NYSE_OPEN_E2_RR1.5_CB1_ATR_P70    NYSE_OPEN     ATR_P70 473.0000 480.0000    -7.0000          1.0148
     GC_NYSE_OPEN_E2_RR2.0_CB1_ATR_P70    NYSE_OPEN     ATR_P70 473.0000 480.0000    -7.0000          1.0148
   GC_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10    NYSE_OPEN   OVNRNG_10 595.0000 589.0000     6.0000          0.9899
   GC_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_10    NYSE_OPEN   OVNRNG_10 595.0000 589.0000     6.0000          0.9899
   GC_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_10    NYSE_OPEN   OVNRNG_10 595.0000 589.0000     6.0000          0.9899
   GC_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50    NYSE_OPEN   OVNRNG_50  55.0000  52.0000     3.0000          0.9455
  GC_US_DATA_1000_E2_RR1.0_CB1_ATR_P50 US_DATA_1000     ATR_P50 725.0000 702.0000    23.0000          0.9683
  GC_US_DATA_1000_E2_RR1.0_CB1_ATR_P70 US_DATA_1000     ATR_P70 473.0000 480.0000    -7.0000          1.0148
   GC_US_DATA_1000_E2_RR1.0_CB1_ORB_G5 US_DATA_1000      ORB_G5 321.0000 331.0000   -10.0000          1.0312
GC_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10 US_DATA_1000   OVNRNG_10 595.0000 589.0000     6.0000          0.9899
 GC_US_DATA_1000_E2_RR1.0_CB1_PDR_R080 US_DATA_1000    PDR_R080 669.0000 665.0000     4.0000          0.9940
 GC_US_DATA_830_E2_RR1.0_CB1_OVNRNG_10  US_DATA_830   OVNRNG_10 595.0000 589.0000     6.0000          0.9899
```

## Finding 3 — The main break is payoff compression on MGC 5-minute trades

At the same 5-minute broad baseline, `MGC` usually keeps broadly similar loss size but materially smaller winners, with some sessions also losing win rate. That compresses expectancy from positive `GC` cells into flat or negative `MGC` cells.

```text
symbol      orb_label rr_target   n win_rate avg_win_r avg_loss_r   avg_r
    GC     CME_REOPEN    1.0000 821   0.3581    0.8841    -1.0000  0.0315
   MGC     CME_REOPEN    1.0000 788   0.3223    0.5600    -0.9872 -0.1402
    GC     CME_REOPEN    1.5000 821   0.2375    1.3572    -1.0000 -0.0282
   MGC     CME_REOPEN    1.5000 788   0.1992    0.9493    -1.0000 -0.2517
    GC     CME_REOPEN    2.0000 821   0.1717    1.8269    -1.0000 -0.1043
   MGC     CME_REOPEN    2.0000 788   0.1472    1.3328    -1.0000 -0.2953
    GC   COMEX_SETTLE    1.0000 914   0.5481    0.8917    -1.0000  0.0721
   MGC   COMEX_SETTLE    1.0000 916   0.5404    0.4917    -0.9888 -0.1589
    GC   COMEX_SETTLE    1.5000 914   0.3993    1.3619    -1.0000  0.0178
   MGC   COMEX_SETTLE    1.5000 916   0.3996    0.8549    -0.9992 -0.1980
    GC   COMEX_SETTLE    2.0000 914   0.3118    1.8363    -1.0000 -0.0094
   MGC   COMEX_SETTLE    2.0000 916   0.3090    1.2364    -0.9992 -0.2257
    GC    EUROPE_FLOW    1.0000 917   0.5823    0.8966    -1.0000  0.1045
   MGC    EUROPE_FLOW    1.0000 917   0.5769    0.5016    -0.9904 -0.1297
    GC    EUROPE_FLOW    1.5000 917   0.4613    1.3704    -1.0000  0.0935
   MGC    EUROPE_FLOW    1.5000 917   0.4689    0.8696    -1.0000 -0.1233
    GC    EUROPE_FLOW    2.0000 917   0.3893    1.8457    -1.0000  0.1079
   MGC    EUROPE_FLOW    2.0000 917   0.3871    1.2558    -1.0000 -0.1267
    GC  LONDON_METALS    1.0000 917   0.5529    0.9089    -1.0000  0.0554
   MGC  LONDON_METALS    1.0000 917   0.5605    0.5426    -1.0000 -0.1353
    GC  LONDON_METALS    1.5000 917   0.4373    1.3862    -1.0000  0.0435
   MGC  LONDON_METALS    1.5000 917   0.4395    0.9255    -1.0000 -0.1538
    GC  LONDON_METALS    2.0000 917   0.3719    1.8627    -1.0000  0.0657
   MGC  LONDON_METALS    2.0000 917   0.3773    1.3054    -1.0000 -0.1292
    GC      NYSE_OPEN    1.0000 918   0.5664    0.9482    -1.0000  0.1096
   MGC      NYSE_OPEN    1.0000 918   0.5577    0.7146    -1.0000 -0.0384
    GC      NYSE_OPEN    1.5000 918   0.4379    1.4350    -1.0000  0.0864
   MGC      NYSE_OPEN    1.5000 918   0.4314    1.1408    -1.0000 -0.0602
    GC      NYSE_OPEN    2.0000 918   0.3584    1.9207    -1.0000  0.0944
   MGC      NYSE_OPEN    2.0000 918   0.3497    1.5631    -1.0000 -0.0682
    GC SINGAPORE_OPEN    1.0000 918   0.5599    0.9122    -1.0000  0.0707
   MGC SINGAPORE_OPEN    1.0000 918   0.5599    0.5656    -0.9934 -0.1205
    GC SINGAPORE_OPEN    1.5000 918   0.4368    1.3909    -1.0000  0.0444
   MGC SINGAPORE_OPEN    1.5000 918   0.4434    0.9605    -1.0000 -0.1308
    GC SINGAPORE_OPEN    2.0000 918   0.3638    1.8709    -1.0000  0.0445
   MGC SINGAPORE_OPEN    2.0000 918   0.3725    1.3528    -1.0000 -0.1235
    GC     TOKYO_OPEN    1.0000 918   0.5621    0.8749    -1.0000  0.0539
   MGC     TOKYO_OPEN    1.0000 918   0.5240    0.4571    -0.9512 -0.2133
    GC     TOKYO_OPEN    1.5000 918   0.4412    1.3441    -1.0000  0.0342
   MGC     TOKYO_OPEN    1.5000 918   0.4336    0.7906    -1.0000 -0.2237
    GC     TOKYO_OPEN    2.0000 918   0.3660    1.8097    -1.0000  0.0284
   MGC     TOKYO_OPEN    2.0000 918   0.3540    1.1552    -1.0000 -0.2370
    GC   US_DATA_1000    1.0000 917   0.5420    0.9497    -1.0000  0.1087
   MGC   US_DATA_1000    1.0000 918   0.5316    0.7272    -1.0000 -0.0301
    GC   US_DATA_1000    1.5000 917   0.4057    1.4345    -1.0000  0.0938
   MGC   US_DATA_1000    1.5000 918   0.4009    1.1498    -1.0000 -0.0388
    GC   US_DATA_1000    2.0000 917   0.3108    1.9203    -1.0000  0.0562
   MGC   US_DATA_1000    2.0000 918   0.3039    1.5736    -1.0000 -0.0818
    GC    US_DATA_830    1.0000 917   0.5049    0.9464    -1.0000  0.0455
   MGC    US_DATA_830    1.0000 917   0.4995    0.7157    -1.0000 -0.0863
    GC    US_DATA_830    1.5000 917   0.4024    1.4314    -1.0000  0.0719
   MGC    US_DATA_830    1.5000 917   0.3915    1.1338    -1.0000 -0.0815
    GC    US_DATA_830    2.0000 917   0.3391    1.9152    -1.0000  0.1070
   MGC    US_DATA_830    2.0000 917   0.3272    1.5543    -1.0000 -0.0621
```

Paired same-day `GC` vs `MGC` outcomes on the 5-minute surface confirm the same story: the day-level paths are still highly correlated, but `MGC` carries a persistent negative R-gap.

```text
     orb_label rr_target  n_pairs corr_r gc_avg_r mgc_avg_r avg_gap_r sign_agree
    CME_REOPEN    1.0000      765 0.7458   0.0704   -0.1232    0.2084     0.4967
    CME_REOPEN    1.5000      765 0.7395   0.0117   -0.2349    0.2811     0.4353
    CME_REOPEN    2.0000      765 0.7686  -0.0713   -0.2784    0.2433     0.4248
    CME_REOPEN    2.5000      765 0.8002  -0.1615   -0.3309    0.1847     0.4248
    CME_REOPEN    3.0000      765 0.8084  -0.2400   -0.3547    0.1542     0.4157
    CME_REOPEN    4.0000      765 0.8257  -0.2374   -0.3781    0.1995     0.4157
  COMEX_SETTLE    1.0000      914 0.8630   0.0721   -0.1589    0.2342     0.8961
  COMEX_SETTLE    1.5000      914 0.8905   0.0178   -0.1980    0.2139     0.8709
  COMEX_SETTLE    2.0000      914 0.8906  -0.0094   -0.2257    0.2120     0.8381
  COMEX_SETTLE    2.5000      914 0.9018  -0.0774   -0.2638    0.1840     0.8020
  COMEX_SETTLE    3.0000      914 0.9040  -0.1134   -0.3089    0.1870     0.7757
  COMEX_SETTLE    4.0000      914 0.9205  -0.2757   -0.4012    0.1291     0.7276
   EUROPE_FLOW    1.0000      917 0.8746   0.1045   -0.1297    0.2341     0.9357
   EUROPE_FLOW    1.5000      917 0.8767   0.0935   -0.1233    0.2167     0.9378
   EUROPE_FLOW    2.0000      917 0.8927   0.1079   -0.1267    0.2346     0.9498
   EUROPE_FLOW    2.5000      917 0.9008   0.0639   -0.1229    0.1892     0.9542
   EUROPE_FLOW    3.0000      917 0.9066   0.0519   -0.1443    0.1963     0.9586
   EUROPE_FLOW    4.0000      917 0.8987   0.0449   -0.1474    0.1923     0.9607
 LONDON_METALS    1.0000      917 0.8836   0.0554   -0.1353    0.1907     0.9444
 LONDON_METALS    1.5000      917 0.8938   0.0435   -0.1538    0.1973     0.9477
 LONDON_METALS    2.0000      917 0.9141   0.0657   -0.1292    0.1949     0.9586
 LONDON_METALS    2.5000      917 0.9193   0.0905   -0.1306    0.2212     0.9640
 LONDON_METALS    3.0000      917 0.9222   0.0889   -0.1069    0.1959     0.9662
 LONDON_METALS    4.0000      917 0.9365   0.0477   -0.1291    0.1759     0.9706
     NYSE_OPEN    1.0000      918 0.9165   0.1096   -0.0384    0.1481     0.9532
     NYSE_OPEN    1.5000      918 0.9296   0.0864   -0.0602    0.1455     0.9466
     NYSE_OPEN    2.0000      918 0.9346   0.0944   -0.0682    0.1662     0.9259
     NYSE_OPEN    2.5000      918 0.9337   0.0491   -0.1211    0.1729     0.8889
     NYSE_OPEN    3.0000      918 0.9220   0.0390   -0.1402    0.1761     0.8660
     NYSE_OPEN    4.0000      918 0.9338  -0.0709   -0.2176    0.1384     0.8301
SINGAPORE_OPEN    1.0000      918 0.8794   0.0707   -0.1205    0.1911     0.9368
SINGAPORE_OPEN    1.5000      918 0.9105   0.0444   -0.1308    0.1752     0.9564
SINGAPORE_OPEN    2.0000      918 0.9149   0.0445   -0.1235    0.1680     0.9586
SINGAPORE_OPEN    2.5000      918 0.9351   0.0879   -0.1038    0.1918     0.9706
SINGAPORE_OPEN    3.0000      918 0.9268   0.0990   -0.0857    0.1846     0.9684
SINGAPORE_OPEN    4.0000      918 0.9404   0.0467   -0.1398    0.1902     0.9706
    TOKYO_OPEN    1.0000      918 0.8160   0.0539   -0.2133    0.2672     0.8878
    TOKYO_OPEN    1.5000      918 0.8229   0.0342   -0.2237    0.2579     0.9096
    TOKYO_OPEN    2.0000      918 0.8548   0.0284   -0.2370    0.2654     0.9292
    TOKYO_OPEN    2.5000      918 0.8507   0.0724   -0.2247    0.2971     0.9325
    TOKYO_OPEN    3.0000      918 0.8403   0.0588   -0.2209    0.2797     0.9346
    TOKYO_OPEN    4.0000      918 0.8590   0.0520   -0.2194    0.2714     0.9466
  US_DATA_1000    1.0000      917 0.9055   0.1087   -0.0289    0.1342     0.8986
  US_DATA_1000    1.5000      917 0.9350   0.0938   -0.0376    0.1307     0.8626
  US_DATA_1000    2.0000      917 0.9278   0.0562   -0.0806    0.1391     0.8179
  US_DATA_1000    2.5000      917 0.9440   0.0623   -0.0907    0.1654     0.7884
  US_DATA_1000    3.0000      917 0.9668  -0.0038   -0.1659    0.1508     0.7492
  US_DATA_1000    4.0000      917 0.9687  -0.1706   -0.3161    0.1261     0.6957
   US_DATA_830    1.0000      917 0.8978   0.0455   -0.0863    0.1342     0.8888
   US_DATA_830    1.5000      917 0.9166   0.0719   -0.0815    0.1605     0.8702
   US_DATA_830    2.0000      917 0.9357   0.1070   -0.0621    0.1798     0.8561
   US_DATA_830    2.5000      917 0.9350   0.0696   -0.1147    0.1989     0.8332
   US_DATA_830    3.0000      917 0.9356   0.0353   -0.1397    0.1817     0.8037
   US_DATA_830    4.0000      917 0.9198  -0.0071   -0.2092    0.1731     0.7732
```

## Finding 4 — Transfer is narrow and mostly collapses above RR1.0

The retired `GC` validated rows do not vanish because the filters stop working. They mostly fail because the same filtered `MGC` rows lose enough payoff that only a small low-RR subset stays positive.

```text
                           strategy_id    orb_label rr_target filter_type     n_gc avg_r_gc t_stat_gc    n_mgc avg_r_mgc t_stat_mgc  same_sign  mgc_positive
 GC_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_50  EUROPE_FLOW    1.0000   OVNRNG_50  55.0000   0.2125    1.6386  52.0000    0.0356     0.2861       True          True
     GC_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50    NYSE_OPEN    1.0000     ATR_P50 646.0000   0.1093    2.8730 627.0000   -0.0161    -0.4675      False         False
     GC_NYSE_OPEN_E2_RR1.0_CB1_ATR_P70    NYSE_OPEN    1.0000     ATR_P70 422.0000   0.1194    2.5325 428.0000   -0.0048    -0.1141      False         False
   GC_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10    NYSE_OPEN    1.0000   OVNRNG_10 595.0000   0.1210    3.0502 589.0000   -0.0200    -0.5551      False         False
   GC_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50    NYSE_OPEN    1.0000   OVNRNG_50  55.0000   0.2216    1.6959  52.0000    0.1468     1.1538       True          True
     GC_NYSE_OPEN_E2_RR1.5_CB1_ATR_P50    NYSE_OPEN    1.5000     ATR_P50 646.0000   0.0688    1.4431 627.0000   -0.0598    -1.3905      False         False
     GC_NYSE_OPEN_E2_RR1.5_CB1_ATR_P70    NYSE_OPEN    1.5000     ATR_P70 422.0000   0.0897    1.5140 428.0000   -0.0551    -1.0474      False         False
   GC_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_10    NYSE_OPEN    1.5000   OVNRNG_10 595.0000   0.1025    2.0545 589.0000   -0.0415    -0.9262      False         False
     GC_NYSE_OPEN_E2_RR2.0_CB1_ATR_P50    NYSE_OPEN    2.0000     ATR_P50 646.0000   0.0922    1.6539 627.0000   -0.0393    -0.7833      False         False
     GC_NYSE_OPEN_E2_RR2.0_CB1_ATR_P70    NYSE_OPEN    2.0000     ATR_P70 422.0000   0.1014    1.4652 428.0000   -0.0599    -0.9810      False         False
   GC_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_10    NYSE_OPEN    2.0000   OVNRNG_10 595.0000   0.1218    2.0847 589.0000   -0.0322    -0.6176      False         False
  GC_US_DATA_1000_E2_RR1.0_CB1_ATR_P50 US_DATA_1000    1.0000     ATR_P50 645.0000   0.1305    3.4329 627.0000   -0.0098    -0.2825      False         False
  GC_US_DATA_1000_E2_RR1.0_CB1_ATR_P70 US_DATA_1000    1.0000     ATR_P70 422.0000   0.1506    3.2085 428.0000    0.0008     0.0188       True          True
   GC_US_DATA_1000_E2_RR1.0_CB1_ORB_G5 US_DATA_1000    1.0000      ORB_G5 320.0000   0.1759    3.2376 331.0000    0.0807     1.6005       True          True
GC_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10 US_DATA_1000    1.0000   OVNRNG_10 595.0000   0.1416    3.5763 589.0000    0.0194     0.5383       True          True
 GC_US_DATA_1000_E2_RR1.0_CB1_PDR_R080 US_DATA_1000    1.0000    PDR_R080 574.0000   0.1491    3.7116 573.0000   -0.0050    -0.1374      False         False
 GC_US_DATA_830_E2_RR1.0_CB1_OVNRNG_10  US_DATA_830    1.0000   OVNRNG_10 594.0000   0.0476    1.1896 589.0000   -0.0726    -2.0141      False         False
```

High-level summary:

- retired `GC` price-safe rows audited: 17
- rows still positive on `MGC` overlap: 5
- rows still positive on `MGC` with `RR > 1.0`: 0
- the warmest surviving bridge is `US_DATA_1000 / ORB_G5 / RR1.0`, but it is still much weaker on `MGC` than on `GC`

## Finding 5 — Do not generalize this beyond the current surface

`MGC` has 15-minute and 30-minute canonical rows in the overlap era, but the `GC` proxy surface here does not. That means there is no honest `GC -> MGC` statement yet for 15m/30m. The translation question proven here is the 5-minute path only.

Minute coverage:

```text
symbol  orb_minutes     n
    GC            5 48942
   MGC            5 48762
   MGC           15 47778
   MGC           30 46020
```

## Bottom Line

The correct conclusion is not "GC proxy was fake" and not "GC edge transfers fine." The honest conclusion is narrower:

- `GC` overlap-era strength is real
- price-safe triggers transfer cleanly enough
- the bridge breaks mainly in 5-minute `MGC` payoff translation
- transfer is mostly too weak above `RR=1.0` to rescue the old `GC` proxy shelf

## Next Action

Run a narrow **MGC 5-minute payoff-compression audit** on the warm translated families (`US_DATA_1000`, `NYSE_OPEN`, `EUROPE_FLOW`) to test whether the right rescue question is lower-RR/exit-shape handling rather than more proxy discovery. Do not reopen broad `GC` proxy exploration before that.
