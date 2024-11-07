---
layout: post
title:  "Prithvi WxC: Foundation Model for Weather and Climate"
date:   2024-11-07 11:29:00 +0800
tags:   Climate
---

> Contribution

+ Despite the mirroring successes of large AI model in both computer science and nature language process, applications of the foundation model principle to atmospheric sciences have been rare.
+ There are various tasks in the atmospheric sciences and author use Prithvi WxC to solve them without designing model for special tasks.

> Method

+ There are several considerations that make masking attractive for pretraining:

  + While both NWP as well as reanalysis data is gridded and dense, observational data is ungridded and sparse. It might not be surprising that the models use masking method to work on observation data.
  + There are use cases such as downscaling or data assimilation when the time step is meaningful. Thus, a foundation model aiming to address all the cases should use the masking for pretraining.
  + On the more technical side, a common problem is the size of the data. Masking is highly memory efficient.

+ In light of the forecast emulators ,which get the performance of a persistence forecast for free by not predictng $X_{t+\delta t}$ but the tendency $X_{t+\delta t}-X_t$, Prithvi WxC model the deviation from historical climate $C_t$ at the certain time.
  $$
  \frac{\hat X_{t+\delta t}-C_{t+\delta t}}{\sigma_C}=f_{\theta}\left[M_{0.5}\left(\frac{X_t-\mu}{\sigma},\frac{X_{t-\delta\tau}-\mu}{\sigma}\right);\frac{C_{t+\delta t}-\mu}{\sigma},S,\delta t, \delta \tau\right]
  $$
  Here, $\mu$ and $\sigma$ are per parameter means and standard deviations. $\sigma ^2_C=\sigma^2_C(X_t-C_t)$ is the variance of the historical anomaly. $S$ are static inputs and $\delta t$ and $\delta \tau$ are the time steps for the target and the inputs respectively.

> Data

+ MERRA-2 (The Modern-Era Retrospective Analysis for Research and Applications Version 2) serves as the primary dataset. Using data from 1980 to 2019 for training and using data from 2020 to 2023 for validating.
+ The climatology $C_t$​ is computed from 20 years of MERRA-2 data following the methodology of the ERA-Interim climatology.

> Architecture

+ Prithvi WxC is a scalable and flexible 2D vision transformer. To keep it as flexible as possible, author aim not to use architecture elements that restrict to "rectangular" topologies for data. 

![Prithvi WxC core architecture](C:\Users\Administrator\Desktop\Prithvi WxC core architecture.png)

+ The data can take the shape (windows, tokens, features). The model alternates attention within a windows and across windows by transposing the window and token dimension between transformer layers.
