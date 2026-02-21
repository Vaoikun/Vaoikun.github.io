---
title: "Tutorial"
---



[Home](https://vaoikun.github.io/)

# Naive Bayes Classifiers Tutorial

## Headline　 <div style="text-align: right;">   <img src="Spam.jpeg" alt="Description" width="100"> </div>

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

After many lines of math, we get:

$$
\arg\max_{k \in K} P(C = k \mid \mathbf{x}) = 
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
```

We then create a method called fit that compute 
$$P(C = \text{Spam}), P(C = \text{Ham})$$ and $$P(x_i|C)$$ 
to fit the model,

```python
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        N = len(y)
        self.ham_prob = sum(y == "ham") / N
        self.spam_prob = sum(y == "spam") / N
        # Tokenization
        tokens = X.astype(str).str.split() # Split at white spaces
        spam_tokens = [w for msg in tokens[y == "spam"] for w in msg]
        ham_tokens  = [w for msg in tokens[y == "ham"]  for w in msg]
        spam_counts = Counter(spam_tokens)
        ham_counts  = Counter(ham_tokens)
        vocab = set(spam_counts) | set(ham_counts)
        V = len(vocab)
        # Total number of word occurrences in each class
        total_spam_words = sum(spam_counts.values())
        total_ham_words  = sum(ham_counts.values())
        self.spam_probs = {}
        self.ham_probs = {}
        for w in vocab:
            self.spam_probs[w] = (spam_counts.get(w, 0) + 1) / (total_spam_words + 2)
            self.ham_probs[w]  = (ham_counts.get(w, 0)  + 1) / (total_ham_words  + 2)

        return self
```

We can see that ```self.ham_probs['out']``` will give the value for $$P(x_i = '\text{out}' \mid C = \text{ham})$$,

```python
# Example model trained on the first 300 data points
nb = NaiveBayesFilter()
nb.fit(X[:300], y[:300])

# Check spam and ham probabilities of 'out'
nb.ham_probs['out']
0.003147128245476003
nb.spam_probs['out']
0.004166666666666667
```

Now that we have trained our model, we can predict the class of a message by calculating

$$
P(C = k) \Pi^n_{i=1} P(x_i \mid C=k) 
$$

for each class $$k$$ .
Directly computing this probability as a product can lead to an issue: underflow. If $$\mathbf{x}$$ is a particularly long message, then, since we are multiplying lots of numbers between 0 and 1, it is possible for the computed probability to underflow, or become too small to be machine representable with ordinary floating-point numbers. In this case the computed probability becomes 0. This is particularly problematic because if underflow happens for a sample for one class, it will likely also happen for all of the other classes, making such samples impossible to classify. To avoid this issue, we will work with the logarithm of the probability. 

```python
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        msgs = X.astype(str).values
        N = len(msgs)
        # log priors
        log_ham_prior = np.log(self.ham_prob)
        log_spam_prior = np.log(self.spam_prob)
        log_unseen = np.log(0.5)
        out = np.zeros((N, 2), dtype=float)
        for i, msg in enumerate(msgs):
            log_ham = log_ham_prior
            log_spam = log_spam_prior
            # tokenization
            for w in msg.split():
                log_ham += np.log(self.ham_probs.get(w, 0.5))
                log_spam += np.log(self.spam_probs.get(w, 0.5))

            out[i, 0] = log_ham
            out[i, 1] = log_spam

        return out
```

This will produce something like,

```python
nb.predict_proba(X[800:805])
array([[ -30.8951931 ,  -35.42406687],
       [-108.85464069,  -91.70332157],
       [ -74.65014875,  -88.71184709],
       [-164.94297917, -133.8497405 ],
       [-127.17743715, -101.32098062]])
```

Finally, we will implement the method ```predict()``` that makes predicted classification,

```python
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        logj = self.predict_proba(X)
        preds = np.where(logj[:, 1] > logj[:, 0], "spam", "ham")
        return preds
```

We can now test our spam filter. We will use the sklearn’s train_test_split function with the default parameters to split the data into training and test sets. Train a NaiveBayesFilter on the train set, and have it predict the labels of each message in the test set. 

```python
X = df.Message
y = df.Label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
nb = NaiveBayesFilter()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
# spam correctly identified
spam_mask = (y_test.to_numpy() == "spam")
spam_correct = np.mean(y_pred[spam_mask] == "spam")
# ham incorrectly identified
ham_mask = (y_test.to_numpy() == "ham")
ham_incorrect = np.mean(y_pred[ham_mask] == "spam")
print("Spam correctly identified:", spam_correct)
print("Ham incorrectly identified:", ham_incorrect)
```

This will yield,

```
Spam correctly identified: np.float64(0.9513513513513514)
Ham incorrectly identified: np.float64(0.012417218543046357)
```

## CTA
While naïve Bayes classifiers are most easily seen as applicable in cases where the features have, ostensibly, well-defined probability distributions, they are applicable in many other cases. In this tutorial, we will apply them to the problem of spam filtering. While it is generally a bad idea to assume independence, naïve Bayes classifiers can still be very effective, even when we are confident that features are not independent.

This is just one of many implementations of the Bayes Naive Classifier. Try implementing your own Bayes Naive Classifier in your machine and try using a different dataset!
You can find similar datasets in websites like [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and [IEEE DataPort](https://ieee-dataport.org/documents/sms-spam-dataset) for free. 
