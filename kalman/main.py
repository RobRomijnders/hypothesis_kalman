from kalman.util import get_dataset
from pylds.models import DefaultLDS
import numpy as np
from scipy.special import logsumexp


# Hypothesis 1: all the data comes from the same data source (1)
# Hypothesis 2: the first half comes from data source (1), the second half comes from data source (2)

# TODO (rob) make relative directory with sym-link
# Data option 1: same data source (1)
# Data option 2: two different data sources (1, 2)
num_samples_half = 500
# Download data from https://www.cs.ucr.edu/~eamonn/time_series_data/
data = get_dataset('../data/UCR_TS_Archive_2015',
                   data_option=2,
                   num_samples=num_samples_half)
num_samples = 2 * num_samples_half


num_mc_samples = 10

"""Model selection / Hypothesis tesing"""
# Now calculate
# log posterior_odds = log prior_odds + log likelihood_ratio
#  - log posterior_odds         = log p(H1|DI)/P(H2|DI)
#  - log prior_odds             = log P(H1|I)/P(H2I)
#  - log likelihood_ratio       = log P(D|H1I)/P(D|H2I)
# The variable I represents all our background information

# Let's assume our prior belief in both hypotheses is equal: P(H1|I) = P(H2I)
# Now log_prior_odds is then log(1) = 0
log_prior_odds = 0

# Calculate log P(D|HI) by integrating out theta in p(Dtheta|HI)=p(D|thetaHI)p(theta|HI)
print('Hypothesis 1')
model = DefaultLDS(D_obs=1, D_latent=2)

log_p_D_given_H1I = []
for _ in range(num_mc_samples):
    model.resample_parameters()
    log_p_D_given_H1I.append(np.sum([model.log_likelihood(np.expand_dims(data[n], 1)) for n in range(num_samples)]))
# In the next line, we do a log-sum-exp over our list.
#  - The outer log puts the evidence on log scale
#  - The sum is over the MC samples
#  - The exp cancels the log in the distribution.logpdf()
log_p_D_given_H1I = logsumexp(log_p_D_given_H1I) - np.log(num_mc_samples)

# Calculate log P(D|H2I)
print('Hypothesis 2')
model1 = DefaultLDS(D_obs=1, D_latent=2)
model2 = DefaultLDS(D_obs=1, D_latent=2)

log_p_D_given_H2I = []
for i in range(num_mc_samples):
    model1.resample_parameters()
    model2.resample_parameters()
    log_p_D_given_H2I.append(
        np.sum([model1.log_likelihood(np.expand_dims(data[n], 1)) for n in range(num_samples_half)]) +
        np.sum([model2.log_likelihood(np.expand_dims(data[n], 1)) for n in range(num_samples_half, num_samples)]))
# In the next line, we do a log-sum-exp over our list.
#  - The outer log puts the evidence on log scale
#  - The sum is over the MC samples
#  - The exp cancels the log in the distribution.logpdf()
log_p_D_given_H2I = logsumexp(log_p_D_given_H2I) - np.log(num_mc_samples)


# Calculate the log_likelihood ratio
log_likelihood_ratio = log_p_D_given_H1I - log_p_D_given_H2I

# So to conclude, our log posterior odds is
log_posterior_odds = log_prior_odds + log_likelihood_ratio

print(f'Hypothesis 1 is {10 * (log_posterior_odds / np.log(10))}dB more likely than hypothesis 2')

print(f'In other words, Hypothesis 1 is {np.exp(log_posterior_odds)} more likely than hypothesis 2')
