---
title: "Tutorial"
---

### Naive Bayes Classifiers Tutorial

## What are Naive Bayes Classifiers?
Naïve Bayes classifiers are a family of machine learning classification methods that use Bayes’ theorem to probabilistically categorize data. They are called naïve because they assume independence between the features. The main idea is to use Bayes’ theorem to determine the probability that a certain data point belongs in a certain class, given the features of that data. Despite what the name may suggest, the naïve Bayes classifier is not a Bayesian method, as it is based on likelihood rather than Bayesian inference.

## What are some applications?
While naïve Bayes classifiers are most easily seen as applicable in cases where the features have, ostensibly, well-defined probability distributions, they are applicable in many other cases. In this tutorial, we will apply them to the problem of spam filtering. While it is generally a bad idea to assume independence, naïve Bayes classifiers can still be very effective, even when we are confident that features are not independent.

## How does it work?
Given the feature vector of a piece of data we want to classify, we want to know which of all the classes is most likely. Essentially, we want to answer the following question,

$$
\arg\max_{k \in K} P(C = k \mid \mathbf{x})
$$

where $$C$$ is the random variable representing the class of data. Using Bayes’ Theorem, we can reformulate this problem into something that is actually computable. 

For any $$k\in K$$,

$$
P(C = k \mid \mathbf{x}) = \frac{P(C = k)P(\mathbf{x} \mid C = k)}{P(\mathbf{x})}
$$

By doing many lines of math, we eventually get,

$$
\arg\max_{k \in K} P(C = k \mid \mathbf{x}) = \arg\max_{k \in K} P(C = k) \Pi^n_{i=1}P(x_i \mid C = k)
$$

