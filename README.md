# MathematicsCourseforDataScienceProbability

## Required


## Description


## Table of Contents (Optional)

Probability for Data Science.



   * [Uncertainty and probability](#uncertainty-and-probability)
      * [What is probability?](#descriptive-statistics-vs-statistics-inference)
      * [Probability in machine learning](#probability-in-machine-learning)
   * [Basics of probability](#basics-of-probability)
      * [Types of probability](#types-of-probability)
      * [Probability Calculation Examples](#probability-calculation-examples)        
      * [Advanced examples with probability](#advanced-examples-with-probability)  
   * [Probability distributions](#probability-distributions)
      * [What is a distribution?](#what-is-a-distribution)
      * [Discrete distributions](#discrete-distributions)
      * [Using the binomial distribution](#using-the-binomial-distribution) 
      * [Continuous distributions](#continuous-distributions)  
      * [How to estimate a distribution?](#how-to-estimate-a-distribution)   
   * [MLE (Maximum Likelihood Estimation)](#MLE-Maximum-likelihood-estimation) 
      * [What is MLE?](#what-is-mle)
      * [MLE in machine learning](#mle-in-machine-learning)
      * [Logistic regression](#logistic-regression)
      * [Logistic Regression Application](#logistic-regression-application)
   * [Bayesian inference](#bayesian-inference)     
      * [Bayes theorem](#bayes-theorem)
      * [Bayes in machine learning](#bayes-in-machine-learning)
      * [Final challenges](#final-challenges)


Uncertainty and probability
============


What is probability?
-----------

Probability is a belief we have about the occurrence of elementary events.

In what cases do we use probability?

Intuitively, we make estimates of the probability of something happening or not, to the unknown
that we have about the relevant information of an event we call it uncertainty.

Chance in these terms does not exist, it represents the absence of knowledge of all
the variables that make up a system.

In other words, probability is a language that allows us to quantify uncertainty.

1. AXIOMS:
It is a set of statements that are not derivable from something more fundamental. we take them for truth
and do not require proof.

   * “Sometimes axioms are compared to seeds, because from them all theory arises”

2. AXIOMS ​​OF PROBABILITY:

The probability is given by the number of success cases over the total (theoretical) number of cases.

P = #-Successful Cases/ # Total-Cases.

Elemental event: It is a single occurrence, “You only have one side of the coin as a result”

Events: These are the possibilities that we have in the system. It is composed of elementary events,
for example, “The result of rolling a die is even”, there are three events (2,4,6) that make up this statement.

Two schools of thought diverge from the interpretation of the above axiom. Frequentist and Bayesian

Example: “I only have two possible outcomes when tossing a coin, 50% chance for each head
, (1/2 and 1/2), if I toss the coin n times, the coin does not land half the time on one side, and then the other”

This equiprobability of occurrence in a sample space occurs under the assumption that
the ratio of successes/totals tends to a p-value. In other words, just tossing the coin
infinite times we can notice that the value of the probability is close to (1/2 or 50%).

3. frequentist school

“Every random variable is described by the sample space that contains all possible events
of that random problem.”

The probability that is assigned as a value to each possible event has several properties to hold.

AXIOM PROPERTIES:

* 0 <= P <= 1
* Certainty: P = 1
* Impossibility P = 0
* Disjunction P(AuB) = P(A) +P(B)


Probability in machine learning
-----------

What are the sources of uncertainty?

* Data: Because our measurement instruments have a margin of error, imperfect and incomplete data are presented, therefore there is uncertainty in the data.
* Model attributes: These are variables that represent a reduced subset of the entire reality of the problem, these variables come from the data and therefore present a certain degree of uncertainty.
* Architecture of the model: A model in math is a simplified representation of reality and being so, by construction, it induces another layer of uncertainty, since being a simplified representation much less information is considered.

And of course, all this uncertainty can be quantified with probability:

Example, a text document classifier:

ima 1

Then, the model will assign a certain probability to each document and thus determine the classification of the documents.

But how does our classification model work inside?

ima 2

So, where does probability apply?

Well, actually not all probabilistic models, when designing it we choose if we want it to be a probabilistic model or not.

For example, if we choose the Naive Vayes model, after we choose the design we now define the training and this is basically that the model learns the concept of probability distribution and it is a way that I use to know what probabilities I assign to one of the possible occurrences of my data, hence the MLE scheme, which is the maximum likelihood estimator and after this is the calibration, the hyper-parameters are configured, this is understood more in artificial neural networks where the number of neurons of a layer It has 10 neurons and each one has its own weights that connect the neurons, so we can calibrate those weights so that the model is smaller and smaller. However, there are parameters that are outside the model and cannot be calibrated, and we call these parameters hyper-parameters, because they are outside the entire optimization scheme. At the end the optimization of the hyper parameters is done. And in the end we have the interpretation, to interpret there are times that you have to know how the model works and apply statistical concepts to be able to interpret it.
