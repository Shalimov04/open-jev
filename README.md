# Open-Jev

## Results

<!-- results -->
| task | type | K | lang | n_train | student acc / F1 | teacher acc / F1 | agree | ECE raw→cal | Brier | GPU p50 ms | ex/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agnews | choice | 4 | en | 4000 | 0.879 / 0.880 | 0.880 / 0.881 | 0.938 | 0.071→0.042 | 0.198 | 20.7 | 382 |

- **agnews**: jhu-clsp/mmBERT-small distilled from the teacher with zero gold labels; argmax accuracy vs gold on 2000 eval examples; calibrated to gold (T=0.74).
<!-- results -->
