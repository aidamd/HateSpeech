import pandas as pd

print("Calculating false negative rates")
biased = pd.read_csv("biased/became_nonhate_SGT.csv")
unbiased = pd.read_csv("unbiased/became_nonhate_SGT.csv")

samples = pd.read_csv("biased/predictions.csv")
print(samples.shape)
num_posts = samples.shape[0] / len(set(samples["SGT"]))
print(num_posts, "samples in the test set")

biased_fp = biased["Frequency"] / num_posts
unbiased_fp = unbiased["Frequency"] / num_posts

print(sum(biased_fp) / len(biased_fp))
print(sum(unbiased_fp) / len(unbiased_fp))


print("Calculating false positive rates")
biased = pd.read_csv("biased/became_hate_SGT.csv")
unbiased = pd.read_csv("unbiased/became_hate_SGT.csv")

samples = pd.read_csv("biased/predictions.csv")
num_posts = samples.shape[0] / len(set(samples["SGT"]))
print(num_posts, "samples in the test set")

biased_fn = biased["Frequency"] / num_posts
unbiased_fn = unbiased["Frequency"] / num_posts

print(sum(biased_fn) / len(biased_fn))
print(sum(unbiased_fn) / len(unbiased_fn))