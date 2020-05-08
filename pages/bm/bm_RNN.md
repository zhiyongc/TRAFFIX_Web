---
title: Page without footer
subtitle: Demo page without footer
layout: page
show_sidebar: false
menubar: benchmark_menu
hide_footer: true
hide_hero: true
---

## Benchsmarks

| Dataset   | Model  | LR       | Optm | Epoch | MAE  | MAPE  | RMSE |
|-----------|--------|----------|------|-------|------|-------|------|
| PEMS-BAY  | LSTM   | 1.00E-03 | Adam | 41    | 1.18 | 2.32% | 1.94 |
|           | GRU    | 1.00E-03 | Adam | 30    | 1.11 | 2.11% | 1.78 |
|           | <span style="color:blue">GCLSTM</span>| 1.00E-02 | Adam | 23    | 0.95 | 1.81% | 1.64 |
| METR-LA   | <span style="color:blue">LSTM</span>  | 1.00E-03 | Adam | 56    | 2.76 | 6.18% | 4.86 |
|           | GRU    | 1.00E-03 | Adam | 35    | 2.83 | 6.44% | 4.93 |
|           | GCLSTM | 1.00E-02 | Adam | 49    | 2.74 | 6.17% | 5.02 |
| LOOP-SEA  | <span style="color:blue">LSTM</span>   | 1.00E-03 | Adam | 39    | 2.43 | 5.78% | 3.63 |
|           | GRU    | 1.00E-03 | Adam | 30    | 2.45 | 5.84% | 3.64 |
|           | GCLSTM | 1.00E-02 | Adam | 22    | 2.53 | 6.08% | 3.83 |
| INRIX-SEA | LSTM   | 1.00E-04 | Adam | 53    | 0.87 | 3.10% | 1.86 |
|           | GRU    | 1.00E-03 | Adam | 31    | 0.88 | 3.18% | 1.84 |
|           | <span style="color:blue">GCLSTM</span> | 1.00E-02 | Adam | 14    | 0.77 | 2.74% | 1.75 |