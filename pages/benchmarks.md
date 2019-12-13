---
title: Benchmarks
subtitle: Performance of various deep learning-based traffic forecasting models
hero_image: ../../img/SeattleTraffic_2.PNG
layout: page
show_sidebar: false
menubar: benchmark_menu
mathjax: true
hide_footer: true
hide_hero: true
<!-- hide_hero: true -->
---


## <span style="font-family:Courier; font-size:1em;">Benchmarks</span>

### <span style="font-family:Courier; font-size:1em;">Mean Absolute Error (MAE)</span>


\begin{equation}
    MAE = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|
\end{equation}

### <span style="font-family:Courier; font-size:1em;">Mean Absolute Percentile Error (MAPE)</span>

\begin{equation}
    MAPE = \frac{1}{N}\sum_{i=1}^N|\frac{y_i - \hat{y}_i}{y_i}|
\end{equation}


### <span style="font-family:Courier; font-size:1em;">Root Mean Square Error (RMSE)</span>

\begin{equation}
    RMSE = (\frac{1}{N} \sum_{i=1}^N {|y_i - \hat{y}_i|}^2 )^{\frac{1}{2}}
\end{equation}
