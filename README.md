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
Impossibility P = 0
Disjunction P(AuB) = P(A) +P(B)


Probability in machine learning
-----------

