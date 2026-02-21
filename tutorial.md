---
title: "Tutorial"
---

# Naive Bayes Classifiers Tutorial

## Headline

Naïve Bayes classifiers are a family of machine learning classification methods that use Bayes’ theorem to probabilistically categorize data.  
They are called naïve because they assume independence between the features. The main idea is to use Bayes’ theorem to determine the probability that a certain data point belongs in a certain class, given the features of that data. Despite what the name may suggest, the naïve Bayes classifier is not a Bayesian method, as it is based on likelihood rather than Bayesian inference.
 
## Introduction

### How does it work?

Given the feature vector of a piece of data we want to classify, we want to know which of all the classes is most likely. Essentially, we want to answer the following question:

$$
\arg\max_{k \in K} P(C = k \mid \mathbf{x})
$$

where $C$ is the random variable representing the class of data. Using Bayes’ Theorem, we can reformulate this problem into something that is actually computable.

For any $k \in K$,

$$
P(C = k \mid \mathbf{x}) = \frac{P(C = k)\,P(\mathbf{x} \mid C = k)}{P(\mathbf{x})}.
$$

By doing many lines of math, we eventually get:

$$
\arg\max_{k \in K} P(C = k \mid \mathbf{x})
=
\arg\max_{k \in K} P(C = k)\prod_{i=1}^{n} P(x_i \mid C = k).
$$

---

### Spam Problem

A spam filter is just a special case of a classifier with two classes: spam and not spam (often called *ham*). Spam filtering is a situation where naïve Bayes classifiers perform relatively well.

We will use the SMS spam dataset from  
[Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

After the data is cleaned by converting to lowercase and removing all punctuation, we are ready to start the classification.

Data is expected to look something like,

|   |   |
|---|---|
| ham | go until jurong point crazy available only in... |
| spam | free entry in 2 a wkly comp to win fa cup final tkts 21st may 2005 text fa to... |
| ham | nah i dont think he goes to usf he lives around here though... |
| spam | freemsg hey there darling its been 3 weeks now and no word back...|

---

## Body

### Naive Bayes Class

We first create a Python class called `NaiveBayesFilter` with a constructor:

```python
class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''

    def __init__(self):
        self.ham_prob = None
        self.spam_prob = None
        self.spam_probs = {}
        self.ham_probs = {}
