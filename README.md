# y_intercept
In this project, we have tested a deep learning model called TCN. The deep learning model shows a test loss of -0.025. This loss [1] is specifically designed for avoiding predicting the next day's index change since the mean of returns are deducted. That is, we want to predict the relative movement of the stocks.

In the future, we could add a Mixture-of-experts layer between a FCN to increase the generalization ability of the model.

[1] Cui et al. AlphaEvolve: A Learning Framework to Discover Novel Alphas in Quantitative Investmnent

Please see the test.ipynb to get a better presentation for the codes.
Please run y_intercept.py on a GPU machine to get a faster performance.
