# runs_corr report (2026-11-01 22:18)

- plan: 49 patches over 15 scrolls; 11 skipped (see patches.csv)
- finished runs: 49/49 source runs, baseline yes
- probe: 48 batches x 32 windows per target, eval 16 batches; training 3 epochs at lr 0.0001
- AdamW sqrt(v_hat) < eps on a median 72.6% of coordinates at the end of training (high = AdamW acts like SGD there; gradients are clipped while loss-scaled)

## Probe reliability
split-half reliability of a target mean gradient (median over runs and targets): 0.813 (full-length 0.897); batches needed for 0.9: median 50, 90th percentile 179. targets below 0.5 full reliability: none.
median simple noise scale B_simple: 209 windows

## Directional alignment (source model s, target t != s)
2352 (s, t) pairs; per-batch cos(g_t, G_s) significantly positive (BH q<0.05): 1453, significantly negative: 320.
cos_src_cross same-scroll mean 0.417 (n=240) vs other-scroll 0.225 (n=2112), Mann-Whitney p=3.3e-19
self (split-half) beats every other target in 22/49 runs (self_dominance.csv)

most conflicting pairs:

| source_label | target_label | source_scroll | target_scroll | cos_src_cross | cos_src_disatt | batch_cos_src_mean | batch_q | cos_g_last | dloss_u_last_rel | eval_char_ap |
|---|---|---|---|---|---|---|---|---|---|---|
| w068 | 0846a_20261012073244 | pherc0172 | pherc0846a | -0.764 | -0.829 | -0.549 | 0.000 | -0.702 | -0.788 | 0.448 |
| w068 | fiber_only_20261029d_flatboi_in_2 | pherc0172 | pherc0846a | -0.668 | -0.730 | -0.450 | 0.000 | -0.622 | 0.498 | 0.442 |
| w068 | paris1_fr34_flat | pherc0172 | phercparis1_fr34 | -0.654 | -0.707 | -0.517 | 0.000 | -0.611 | -1.872 | 0.177 |
| w068 | fiber_only_20261029d_flatboi_out_1 | pherc0172 | pherc0846a | -0.643 | -0.713 | -0.401 | 0.000 | -0.595 | 0.013 | 0.539 |
| w068 | paris1_fr34 | pherc0172 | phercparis1_fr34 | -0.637 | -0.691 | -0.441 | 0.000 | -0.538 | -1.945 | 0.241 |
| w068 | 0846a_m28 | pherc0172 | pherc0846a | -0.636 | -0.714 | -0.355 | 0.000 | -0.619 | 0.710 | 0.433 |
| w068 | 0846a_in_1 | pherc0172 | pherc0846a | -0.589 | -0.655 | -0.280 | 0.000 | -0.600 | 0.388 | 0.487 |
| w043 | fiber_only_20261029d_flatboi_in_2 | pherc0139 | pherc0846a | -0.564 | -0.698 | -0.391 | 0.000 | -0.309 | 10.038 | 0.436 |
| w068 | 0846a_out_3 | pherc0172 | pherc0846a | -0.558 | -0.654 | -0.261 | 0.000 | -0.462 | -1.791 | 0.549 |
| w087 | scroll6_fr8_flat | pherc0172 | pherc51cr4_fr8 | -0.546 | -0.876 | -0.373 | 0.000 | -0.120 | 3.199 | 0.377 |
| p841 | 0846a_20261012073244 | pherc0841 | pherc0846a | -0.542 | -0.741 | -0.409 | 0.000 | -0.004 | -2.422 | 0.579 |
| w068 | fiber_only_20261029d_flatboi_in_1 | pherc0172 | pherc0846a | -0.539 | -0.606 | -0.273 | 0.000 | -0.561 | -0.939 | 0.454 |
| w068 | 0846a_out_2 | pherc0172 | pherc0846a | -0.534 | -0.646 | -0.287 | 0.000 | -0.569 | -0.042 | 0.408 |
| w087 | paris2_fr143 | pherc0172 | phercparis2_fr143 | -0.532 | -1.045 | -0.386 | 0.000 | -0.071 | 3.441 | 0.516 |
| w030 | w013 | pherc0139 | pherc1667 | -0.526 | -0.663 | -0.280 | 0.000 | -0.676 | 7.115 | 0.345 |

most aligned cross-patch pairs:

| source_label | target_label | source_scroll | target_scroll | cos_src_cross | cos_src_disatt | batch_cos_src_mean | batch_q | cos_g_last | dloss_u_last_rel | eval_char_ap |
|---|---|---|---|---|---|---|---|---|---|---|
| p9b_487 | p9b_592 | pherc0009b | pherc0009b | 0.909 | 0.982 | 0.395 | 0.000 | 0.391 | -1.348 | 0.594 |
| 0846a_20261012073244 | 0846a_m28 | pherc0846a | pherc0846a | 0.905 | 0.991 | 0.411 | 0.000 | 0.210 | 1.211 | 0.626 |
| p9b_487 | ph1447_w058 | pherc0009b | pherc1447 | 0.898 | 0.975 | 0.438 | 0.000 | 0.431 | -1.845 | 0.604 |
| p9b_487 | w039 | pherc0009b | pherc0139 | 0.893 | 0.962 | 0.450 | 0.000 | 0.528 | -1.828 | 0.253 |
| 0846a_out_2 | 0846a_out_3 | pherc0846a | pherc0846a | 0.879 | 0.981 | 0.399 | 0.000 | 0.757 | -1.178 | 0.567 |
| p9b_487 | w045 | pherc0009b | pherc0139 | 0.879 | 0.937 | 0.527 | 0.000 | 0.389 | -2.800 | 0.382 |
| 0846a_out_1 | fiber_only_20261029d_flatboi_out_2 | pherc0846a | pherc0846a | 0.869 | 0.942 | 0.357 | 0.000 | 0.612 | -3.187 | 0.598 |
| 0846a_20261012073244 | fiber_only_20261029d_flatboi_out_2 | pherc0846a | pherc0846a | 0.863 | 0.934 | 0.450 | 0.000 | 0.178 | 1.348 | 0.571 |
| 0846a_out_2 | 0846a_out_1 | pherc0846a | pherc0846a | 0.863 | 0.987 | 0.372 | 0.000 | 0.722 | -1.029 | 0.564 |
| p9b_487 | p343_letters | pherc0009b | pherc0343 | 0.862 | 0.938 | 0.396 | 0.000 | 0.539 | -0.880 | 0.753 |
| p9b_487 | p841_w00 | pherc0009b | pherc0841 | 0.858 | 0.941 | 0.365 | 0.000 | 0.508 | -1.364 | 0.385 |
| w068 | paris4 | pherc0172 | phercparis4 | 0.855 | 0.959 | 0.437 | 0.000 | 0.657 | 3.151 | 0.452 |
| w044 | w039 | pherc0139 | pherc0139 | 0.854 | 0.991 | 0.439 | 0.000 | 0.149 | -1.485 | 0.314 |
| p9b_487 | 500P2_m2 | pherc0009b | pherc0500p2 | 0.853 | 0.920 | 0.390 | 0.000 | 0.530 | -1.285 | 0.416 |
| p9b_487 | w030 | pherc0009b | pherc0139 | 0.849 | 0.912 | 0.475 | 0.000 | 0.511 | -1.127 | 0.512 |

## Group structure tests (label permutation, one-sided: same group more similar)
| matrix | grouping | n_same | mean_same | mean_diff | contrast | p_perm |
|---|---|---|---|---|---|---|
| consensus_cos_cross | scroll | 120 | 0.591 | 0.380 | 0.211 | 0.000 |
| consensus_cos_cross | regime | 732 | 0.448 | 0.326 | 0.122 | 0.000 |
| sym_cos_src_cross | scroll | 120 | 0.417 | 0.225 | 0.192 | 0.000 |
| sym_cos_src_cross | regime | 732 | 0.272 | 0.199 | 0.073 | 0.013 |
| sym_cos_g_last | scroll | 120 | 0.146 | 0.089 | 0.057 | 0.012 |
| sym_cos_g_last | regime | 732 | 0.118 | 0.057 | 0.062 | 0.003 |
| sym_dot_src_rel | scroll | 120 | 0.861 | -0.320 | 1.181 | 0.038 |
| sym_dot_src_rel | regime | 732 | 0.265 | -0.966 | 1.231 | 0.237 |
| sym_delta_eval_char_ap | scroll | 120 | 0.096 | 0.041 | 0.055 | 0.000 |
| sym_delta_eval_char_ap | regime | 732 | 0.055 | 0.033 | 0.022 | 0.003 |
| sym_delta_eval_pr_auc | scroll | 120 | 0.098 | 0.039 | 0.058 | 0.000 |
| sym_delta_eval_pr_auc | regime | 732 | 0.055 | 0.030 | 0.025 | 0.000 |

## Consistency checks
- asymmetry: Spearman(cos_src_cross[s,t], [t,s]) = 0.003
- gradient proxy vs transfer: Spearman(cos_src_cross, delta_eval_char_ap) = 0.054 (p=0.0087, n=2352)
- gradient proxy vs transfer: Spearman(cos_src_cross, delta_eval_pr_auc) = 0.046 (p=0.024, n=2352)
- gradient proxy vs transfer: Spearman(cos_src_cross, delta_eval_loss) = 0.019 (p=0.36, n=2352)
- gradient proxy vs transfer: Spearman(cos_g_last, delta_eval_char_ap) = 0.045 (p=0.03, n=2352)
- gradient proxy vs transfer: Spearman(cos_g_last, delta_eval_pr_auc) = 0.038 (p=0.067, n=2352)
- gradient proxy vs transfer: Spearman(cos_g_last, delta_eval_loss) = 0.002 (p=0.93, n=2352)
- gradient proxy vs transfer: Spearman(dot_src_rel, delta_eval_char_ap) = -0.055 (p=0.0076, n=2352)
- gradient proxy vs transfer: Spearman(dot_src_rel, delta_eval_pr_auc) = -0.056 (p=0.0069, n=2352)
- gradient proxy vs transfer: Spearman(dot_src_rel, delta_eval_loss) = 0.017 (p=0.4, n=2352)
- gradient proxy vs transfer: Spearman(dloss_u_last_rel, delta_eval_char_ap) = -0.079 (p=0.00013, n=2352)
- gradient proxy vs transfer: Spearman(dloss_u_last_rel, delta_eval_pr_auc) = -0.087 (p=2.4e-05, n=2352)
- gradient proxy vs transfer: Spearman(dloss_u_last_rel, delta_eval_loss) = 0.116 (p=1.7e-08, n=2352)

## Clustering of the consensus affinity (average linkage on (1 - cos) / 2)
| k | sizes | silhouette | silhouette_null_mean | p_null | within_mean | between_mean | bootstrap_ari_mean | bootstrap_ari_p10 | eigengap |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 6/43 | 0.213 | 0.076 | 0.005 | 0.435 | 0.282 | 0.741 | 0.052 | 0.098 |
| 3 | 6/42/1 | 0.158 | 0.066 | 0.005 | 0.441 | 0.286 | 0.639 | 0.299 | 0.084 |
| 4 | 6/11/31/1 | 0.212 | 0.074 | 0.005 | 0.500 | 0.319 | 0.564 | 0.241 | 0.028 |
| 5 | 6/11/2/29/1 | 0.185 | 0.084 | 0.005 | 0.516 | 0.323 | 0.634 | 0.407 | 0.016 |
| 6 | 2/4/11/2/29/1 | 0.191 | 0.094 | 0.005 | 0.519 | 0.324 | 0.696 | 0.299 | 0.010 |
| 7 | 2/4/11/2/2/27/1 | 0.198 | 0.103 | 0.005 | 0.533 | 0.330 | 0.744 | 0.427 | 0.005 |
| 8 | 2/4/11/2/2/3/24/1 | 0.184 | 0.115 | 0.005 | 0.549 | 0.341 | 0.680 | 0.402 | 0.003 |
| 9 | 2/3/1/11/2/2/3/24/1 | 0.177 | 0.126 | 0.005 | 0.549 | 0.342 | 0.660 | 0.450 | 0.005 |
| 10 | 2/3/1/11/2/2/3/14/10/1 | 0.185 | 0.136 | 0.010 | 0.594 | 0.362 | 0.623 | 0.463 | 0.002 |

best silhouette at k=2; members:

- cluster 1: scroll6_fr8 (pherc51cr4_fr8), scroll6_fr8_flat (pherc51cr4_fr8), paris1_fr34 (phercparis1_fr34), paris1_fr34_flat (phercparis1_fr34), paris2_fr143 (phercparis2_fr143), paris2_fr143_flat (phercparis2_fr143)
- cluster 2: p9b_487 (pherc0009b), w030 (pherc0139), w035 (pherc0139), w039 (pherc0139), w040 (pherc0139), w041 (pherc0139), w043 (pherc0139), w044 (pherc0139), w045 (pherc0139), w059 (pherc0139), w068 (pherc0172), w087 (pherc0172), p343 (pherc0343p), p343_letters (pherc0343), 500P2_front (pherc0500p2), 500P2_m1 (pherc0500p2), seg46527 (pherc0814), p841 (pherc0841), p841_w00 (pherc0841), 0846a_20260927134928 (pherc0846a), 0846a_20261012073244 (pherc0846a), 0846a_in_1 (pherc0846a), 0846a_m28 (pherc0846a), 0846a_out_1 (pherc0846a), 0846a_out_2 (pherc0846a), 0846a_out_3 (pherc0846a), ph1447_w058 (pherc1447), ph1447_w060 (pherc1447), cr1fr3 (pherc1667), cr1fr3_flat (pherc1667), w013 (pherc1667), paris2_fr47 (phercparis2_fr143), paris2_fr47_flat (phercparis2_fr143), paris4 (phercparis4), p9b_592 (pherc0009b), title (pherc0139), p343_4b2 (pherc0343p), 500P2_m2 (pherc0500p2), seg0814_459 (pherc0814), fiber_only_20261029d_flatboi_in_1 (pherc0846a), fiber_only_20261029d_flatboi_in_2 (pherc0846a), fiber_only_20261029d_flatboi_out_1 (pherc0846a), fiber_only_20261029d_flatboi_out_2 (pherc0846a)

## Alignment by parameter group (mean cross-half cosine)
| group | n_params | grad_share | self | same_scroll | other_scroll |
|---|---|---|---|---|---|
| enc1 | 31232 | 0.457 | 0.509 | 0.391 | 0.183 |
| early2d_enc2 | 55552 | 0.124 | 0.477 | 0.415 | 0.215 |
| text_region_head | 257 | 0.111 | 0.720 | 0.175 | 0.028 |
| gated_cue_stem | 1176 | 0.086 | 0.559 | 0.379 | 0.237 |
| early2d_head | 33 | 0.050 | 0.715 | 0.543 | 0.486 |
| early2d_enc3 | 221696 | 0.042 | 0.510 | 0.459 | 0.263 |
| early2d_dec1 | 27776 | 0.039 | 0.558 | 0.484 | 0.418 |
| early2d_bottleneck | 885760 | 0.028 | 0.591 | 0.505 | 0.341 |
| early_depth_fuse | 2048 | 0.023 | 0.497 | 0.415 | 0.216 |
| early2d_dec2 | 110848 | 0.013 | 0.526 | 0.491 | 0.391 |
| dual_scale_head | 33 | 0.012 | 0.994 | 0.912 | 0.755 |
| early2d_dec3 | 442880 | 0.006 | 0.502 | 0.508 | 0.372 |
| early2d_up1 | 8224 | 0.004 | 0.624 | 0.549 | 0.467 |
| early2d_up2 | 32832 | 0.002 | 0.589 | 0.545 | 0.428 |
| fiber_coordinate_input | 128 | 0.001 | 0.188 | 0.173 | 0.101 |
| early2d_up3 | 131200 | 0.001 | 0.561 | 0.543 | 0.407 |
| early_depth_attn | 33 | 0.000 | 0.396 | 0.303 | 0.131 |
| new_surface_input | 96 | 0.000 | 0.512 | 0.354 | 0.113 |
| lse_r | 1 | 0.000 | 0.633 | 0.579 | 0.424 |
| dual_scale_local | 30368 | 0.000 | 0.761 | 0.573 | 0.467 |

## Skipped patches
- 20260303000000: missing dilated_inklabels/20260303000000.png
- 20260310000000: missing dilated_inklabels/20260310000000.png
- 20260306000000: missing dilated_inklabels/20260306000000.png
- 20260325000000: missing dilated_inklabels/20260325000000.png
- 20260206000001: missing dilated_inklabels/20260206000001.png
- 20260318000000: missing dilated_inklabels/20260318000000.png
- 20260227000000: missing dilated_inklabels/20260227000000.png
- 20260115000001: missing dilated_inklabels/20260115000001.png
- 20260210000000: missing dilated_inklabels/20260210000000.png; missing surface_labels/20260210000000/depth.npy; missing surface_labels/20260210000000/confidence.npy; missing surface_labels/20260210000000/metadata.json
- 20240304144031: missing /vesuvius/ves_zarrs2/20240304144031.matched.zarr
- 20260930144762: missing train_masks/20260930144762.png
