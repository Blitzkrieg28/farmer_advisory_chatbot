# Vision results (auto-generated 2026-11-13 10:41)

|                                | efficientnet_b0_plantvillage_seed42   | efficientnet_b0_pv_pd_seed42   |
|:-------------------------------|:--------------------------------------|:-------------------------------|
| seed                           | 42                                    | 42                             |
| epochs                         | 20                                    | 20                             |
| best_epoch                     | 17                                    | 10                             |
| best_val_loss                  | 0.0142                                | 0.0116                         |
| final_train_loss_unw           | 0.0008                                | 0.0018                         |
| train_time_min                 | 49.9                                  | 59.5                           |
| PV_val_acc                     | 99.84                                 | 99.81                          |
| PV_test_acc                    | 99.9                                  | 99.71                          |
| PV_test_acc_k_n                | 3125/3128                             | 3119/3128                      |
| PV_test_acc_CI                 | 99.78-100.00                          | 99.52-99.87                    |
| PV_test_f1                     | 99.88                                 | 99.66                          |
| PlantDoc_acc                   | 30.71                                 | 66.14                          |
| PlantDoc_acc_k_n               | 39/127                                | 84/127                         |
| PlantDoc_acc_CI                | 22.8-38.6                             | 58.3-74.0                      |
| PlantDoc_f1                    | 21.83                                 | 63.44                          |
| PlantDoc_f1_CI                 | 15.5-27.4                             | 54.5-69.6                      |
| PlantDoc_crop_level_acc        | 65.4                                  | 92.1                           |
| PlantDoc_crop_known_acc_ORACLE | 40.9                                  | 73.2                           |
| PlantDoc_ECE                   | 0.595                                 | 0.237                          |
| conf_when_correct              | 0.968                                 | 0.934                          |
| conf_when_wrong                | 0.873                                 | 0.768                          |
| params                         | 4025482                               | 4025482                        |
| cpu_latency_ms                 | 35.12                                 | 36.48                          |

## Paired gain on PlantDoc
| run                          | baseline                            |   acc_gain_pts | acc_CI         |   f1_gain_pts | f1_CI          |
|:-----------------------------|:------------------------------------|---------------:|:---------------|--------------:|:---------------|
| efficientnet_b0_pv_pd_seed42 | efficientnet_b0_plantvillage_seed42 |           35.4 | +26.0 to +45.7 |          41.2 | +31.9 to +50.4 |

## Limitations to state
- PlantDoc test has only 127 images (CIs are wide).
- Checkpoint chosen by PlantVillage validation loss (no PlantDoc validation split).
- PlantVillage accuracy is optimistic (lab images, near-duplicates).
- crop-known accuracy is an ORACLE upper bound.
