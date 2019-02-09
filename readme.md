# Model selection for time series 

This project focuses on model selection for time series. We will answer the question "I have two buckets of time series. Are they from the same source? Or are they from different sources?" In all honesty, it took me a year to figure our the methodology for this project. About a year ago, a friend asked me this very question. She indeed had two buckets of time series. She interned at the time in a hospital. One set of time series came from control patients. Another set came from patients with a certain intervention. She needed a statistical conclusion for her publication. "Did the intervention change the time series?"

## Struggle
I struggled with the following part "how to compare likelihoods when models have different number of free parameters?". We can think of many models for time series: HMM's, ARMA, LDS. However, fitting a separate model per bucket will result in a higher likelihood. Always. I struggled to compare the performance of one model with the performance of bigger models. More parameters equals better performance, right?

## Calling in advise
At the GP summer school, I asked the question to quite some experts. None of them gave a resolute answer. However, they highlighted a key insight. Every statistical conclusion has underlying assumptions. One cannot discern two buckets of time series without making assumptions on the data. By fitting an Auto regressive process, we assume that the data is a sequence of correlated progressions. By fitting a Linear dynamical system, we assume that there exist a latent sequence of states and also assumes the observed sequence is stationary. Although some of these models are quite rich, they nevertheless impose assumptions.

## The breakthrough
The breakthrough followed after reading Jaynes' book on probability theory. I struggled with the fact that likelihoods for models with more parameters will necessarily be better. Jaynes puts all of machine learning in a probabilistic context. The likelihood is simply one step in calculating the probability of an hypothesis. As we will later see, we must 
_marginalize out_ the parameters of our model. This marginalization naturally embodies a penalty for complex models. Jaynes explains this in chapter 20

# Model and hypotheses
The model we choose is a Linear Dynamical system (LDS). An LDS is the model behind the famous Kalman filter algorithm. It is generally considered a rich model for time series. Under the hood, it assumes the observed sequence of states is stationary. We cannot confirm the stationarity of our data. Moreover, even though are data is not stationary, the LDS still remains a useful model. 

## Our hypotheses

We define the following hypotheses:

  * hypothesis 1 <img alt="$H_1$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/208fbcc5ce29722c2f701868ac31fc3c.svg" align="middle" width="20.141385pt" height="22.381919999999983pt"/>: both data sets <img alt="$D_1$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/eb4779c5fded13881cb5f169b1f10c73.svg" align="middle" width="20.086935000000004pt" height="22.381919999999983pt"/>, <img alt="$D_2$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/9f0028b414617caf75a357cfb98e7497.svg" align="middle" width="20.086935000000004pt" height="22.381919999999983pt"/> come from the same (unknown) source
  * hypothesis 2 <img alt="$H_2$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/912631c954499428b64ab8d828ac8cb6.svg" align="middle" width="20.141385pt" height="22.381919999999983pt"/>: data set <img alt="$D_1$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/eb4779c5fded13881cb5f169b1f10c73.svg" align="middle" width="20.086935000000004pt" height="22.381919999999983pt"/> comes from one (unknown) source. data set <img alt="$D_2$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/9f0028b414617caf75a357cfb98e7497.svg" align="middle" width="20.086935000000004pt" height="22.381919999999983pt"/> comes from another source
  * All data sources will be modeled a LDS. We take the parameter priors from the default priors in the _pylds_ package

# Inference

In Jaynes' treatment of model comparison, we calculate the probability of a hypothesis. Comparing two models then amounts to comparing the posterior probability for two models. We will look for answers of the form "hypothesis 1 is _x_ more likely than hypothesis 2"

Now we can calculate the probability of an hypothesis being true:

<img alt="$p(H|D) = \frac{p(D|H)p(H)}{p(D)}$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/d17badbe6e53532e9394b12cef908ede.svg" align="middle" width="150.73278pt" height="33.14091000000001pt"/>

Our main interest is the relative probability of an hypothesis being true. Therefore, we divide the probability for hypothesis 1 by the probability for hypothesis 2:

<img alt="$\frac{p(H_1|D)}{p(H_2|D)} = \frac{p(H_1)p(D|H_1)p(D)}{p(H_2)p(D|H_2)p(D)} = \frac{p(H_1|)p(D|H_1)}{p(H_2|)p(D|H_2)}$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/2b1fee0b5ba43d8f8cb1a82749955e45.svg" align="middle" width="299.86572pt" height="33.14091000000001pt"/>

In the code, we will also express this ratio in decibels. Decibels are a little easier to interpret.

<img alt="$10\log_{10} \frac{p(H_1|D)}{p(H_2|D)} =  10\log_{10} \frac{p(H_1)}{p(H_2)} + 10\log_{10} \frac{p(D|H_1)}{p(D|H_2)}$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/bc285be5e53ef59b82e844b26e30738d.svg" align="middle" width="355.534245pt" height="33.14091000000001pt"/>

As we assign equal prior belief to the hypotheses, we focus on defining the likelihood of the data given one of the hypotheses.ke

## Data likelihood

The data likelihood deserves some special attention. It contains the probability of the data given our assumptions. However, we model the data using a set of unknown parameters. (This point is where I struggled a lot for almost a year.)

We are looking for <img alt="$p(D|H)$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/5943436136b2226bb8954effe29f5c69.svg" align="middle" width="54.490919999999996pt" height="24.56552999999997pt"/>. Yet we only have <img alt="$p(D|H\theta)$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/06085758362fdea8958a8d0cf6b0155c.svg" align="middle" width="62.635485pt" height="24.56552999999997pt"/>, where <img alt="$\theta$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.143030500000002pt" height="22.745910000000016pt"/> contains all the parameters in our model. 

The key insight now is to _marginalise out_ the unknown parameters. This marginalisation follows the intuition that _if we don't know a parameter, we must omit it from our reasoning_. 

<img alt="$p(D|H_1) = \int_\theta p(D|H_1\theta)p(\theta|H_1) d\theta$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/0190702315018349c9dedf18c8d40b9e.svg" align="middle" width="240.339495pt" height="26.48447999999999pt"/>

For the LDS, <img alt="$\theta$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.143030500000002pt" height="22.745910000000016pt"/> contains many real valued parameters. It would be hard to make this integration by hand. Therefore, we resort to a Monte Carlo approximation:

<img alt="$E_{ \ \theta \sim p(\theta|H_1)} \ [ \ p(D|H_1\theta) \ ]$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/6b684416c938900cbdbef7a6437c6970.svg" align="middle" width="174.242145pt" height="24.56552999999997pt"/>

For hypothesis 2, we must marginalise out the parameters for a model per each buckets. Let's denote the parameters for model 1 with <img alt="$\theta_1$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/edcbf8dd6dd9743cceeee21183bbc3b6.svg" align="middle" width="14.216235000000003pt" height="22.745910000000016pt"/> and for model 2 with <img alt="$\theta_2$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/f1fe0aebb1c952f09cdbfd83af41f50e.svg" align="middle" width="14.216235000000003pt" height="22.745910000000016pt"/>

<img alt="$p(D|H_2) = \int_{\theta_1, \theta_2} p(D|H_2\theta_1)p(\theta_1|H_2) \ \ p(D|H_2\theta_2)p(\theta_2|H_2) \ \ d\theta_1 d\theta_2$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/a6b6684d1efeae7d8e8c96bfc7615514.svg" align="middle" width="465.56449499999997pt" height="26.48447999999999pt"/>

Note that here we have <img alt="$p(\theta|H_1) = p(\theta_1|H2) = p(\theta_2|H_2)$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/0658a7323ceb784bdb084b33cb18a3c1.svg" align="middle" width="223.70254499999996pt" height="24.56552999999997pt"/>

Likewise, for hypothesis 2 we use the Monte Carlo approximation

<img alt="$E_{ \ \theta_1 \sim p(\theta_1|H_2), \ \theta_2 \sim p(\theta_2|H_2)} \ [ \ p(D|H_2\theta_1) \ p(D|H_2\theta_2) \ ]$" src="https://github.com/RobRomijnders/hypothesis_kalman/blob/master/svgs/a7c288be1cbd705ef0030ed9ce95e463.svg" align="middle" width="357.50269499999996pt" height="24.56552999999997pt"/>


# Results
The results unfortunately contain no fancy plots. We use data from the ECG5000 UCR archive. To test our algorithm both ways, you can generate two buckets from one source or from two sources. 

This would give you the following results:


When generating two buckets from one source:
`Hypothesis 1 is 227909.0746248149dB more likely than hypothesis 2`

When generating two buckets from two sources:
`Hypothesis 1 is -82068.4329797105dB more likely than hypothesis 2`

So we see that the algorithm favors the correct hypothesis in both cases

# Conclusion

This post was the third in a sequence of three posts on model comparison. We see that Jaynes' approach to model selection naturally penalizes complex models. When the data actually demands a complex model, like in our two bucket-two source example, then it favours the more complex model correctly. 

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com
