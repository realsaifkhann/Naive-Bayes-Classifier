# Naive Bayes Classifier

Naive Bayes is a **probabilistic machine learning algorithm** used for **classification tasks**. It is based on **Bayes' Theorem** and assumes that input features are **conditionally independent given the class label**. Despite this simplifying assumption, the algorithm performs well in many real-world applications such as **spam detection, sentiment analysis, and document classification**.

---

## Bayes’ Theorem

The Naive Bayes algorithm is derived from Bayes’ Theorem:

P(C | X) = [ P(X | C) × P(C) ] / P(X)

Where:

* **P(C | X)** → Posterior Probability: Probability of class **C** given the features **X**
* **P(X | C)** → Likelihood: Probability of observing features **X** given class **C**
* **P(C)** → Prior Probability: Initial probability of class **C** before observing the data
* **P(X)** → Evidence: Overall probability of observing the features

The model calculates the posterior probability for each class and predicts the class with the **highest probability**.

---

## Naive Independence Assumption

Naive Bayes assumes that all features are **independent given the class label**.
This simplifies the likelihood calculation as:

P(X | C) = P(x₁ | C) × P(x₂ | C) × ... × P(xₙ | C)

This assumption significantly reduces computational complexity and allows the model to scale efficiently to **high-dimensional datasets**, especially in text classification problems.

---

## Feature Representation for Text Data

For text-based problems, documents must be converted into numerical form. A common approach is the **Bag-of-Words model**, where each unique word becomes a feature and its value represents the **frequency of that word in the document**.

This representation allows machine learning models to process textual information numerically.

---

## Types of Naive Bayes

### Gaussian Naive Bayes

Used when features are **continuous numerical variables**. It assumes the data follows a **normal (Gaussian) distribution**.

### Multinomial Naive Bayes

Primarily used for **text classification**, where features represent **word counts or frequencies**.

### Bernoulli Naive Bayes

Used when features are **binary**, representing the **presence or absence of a feature**.

---

## Laplace Smoothing

A common issue in Naive Bayes is the **zero probability problem**. If a feature never appears in the training data for a given class, the likelihood becomes zero, which causes the entire probability calculation to become zero.

Laplace smoothing resolves this by adding a small constant (usually 1) to each count:

P(xᵢ | C) = ( count(xᵢ , C) + 1 ) / ( total feature count in class C + V )

Where:

* **count(xᵢ , C)** = number of occurrences of feature *xᵢ* in class *C*
* **V** = total number of unique features (vocabulary size)

This ensures that **no probability becomes zero**, making the model more robust.

---

## Advantages

* Computationally **efficient and fast to train**
* Performs well with **high-dimensional data**
* Requires **relatively small training datasets**
* Particularly effective for **text classification problems**

---

## Limitations

* Assumes **feature independence**, which may not always hold in real datasets
* Cannot capture **relationships between features**
* Performance may decline when features are **highly correlated**

---

## Conclusion

Naive Bayes remains a widely used baseline algorithm for classification problems. Its simplicity, efficiency, and strong performance in text-based applications make it an essential algorithm in many machine learning workflows.
