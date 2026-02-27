"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    """
    STEP 1: Consider a standard 52-card deck.
            Assume 4 Aces.

    STEP 2: Compute analytically:
            - P(A)
            - P(B)
            - P(B | A)
            - P(A ∩ B)

    STEP 3: Check independence:
            P(A ∩ B) ?= P(A)P(B)
            
            Two events A and B are independent if and only if:
            P(A ∩ B) = P(A) × P(B)
            If this equality holds, the events are independent (occurrence of one
            does not affect the probability of the other). This test verifies whether
            drawing an Ace on the first draw affects the probability of drawing an Ace
            on the second draw (without replacement).

    STEP 4: Simulate 200,000 experiments
            WITHOUT replacement.
            Use random_state=42.

            Estimate:
            - empirical P(A)
            - empirical P(B | A)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(B | A)
            empirical P(B | A)

    RETURN:
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    """

    # Theoretical probabilities
    P_A = 4 / 52
    
    # Law of Total Probability for P(B):
    # P(B) = P(B|A)P(A) + P(B|A^c)P(A^c) = 4/52
    P_B = 4 / 52
    
    P_B_given_A = 3 / 51
    P_AB = P_A * P_B_given_A
    
    # STEP 3: Check independence by comparing P(A ∩ B) with P(A) × P(B)
    P_A_times_P_B = P_A * P_B
    is_independent = abs(P_AB - P_A_times_P_B) < 1e-10

    # Simulation
    np.random.seed(42)
    trials = 200000
    deck = np.array([1] * 4 + [0] * 48)
    emp_A_count = 0
    emp_B_given_A_count = 0
    emp_A_events = 0
    for _ in range(trials):
        draw = np.random.choice(deck, size=2, replace=False)
        if draw[0] == 1:
            emp_A_count += 1
            emp_A_events += 1
            if draw[1] == 1:
                emp_B_given_A_count += 1
    empirical_P_A = emp_A_count / trials
    empirical_P_B_given_A = emp_B_given_A_count / emp_A_events if emp_A_events > 0 else 0
    absolute_error = abs(empirical_P_B_given_A - P_B_given_A)

    return P_A, P_B, P_B_given_A, P_AB, empirical_P_A, empirical_P_B_given_A, absolute_error


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    """
    STEP 1: Define Bernoulli(p) PMF:
            p_X(x) = p^x (1-p)^(1-x)

    STEP 2: Compute theoretical:
            - P(X = 1)
            - P(X = 0)

    STEP 3: Simulate 100,000 bulbs
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X = 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X = 1)
            empirical P(X = 1)

    RETURN:
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    """

    # Theoretical probabilities
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    # Simulation
    np.random.seed(42)
    trials = 100000
    samples = np.random.choice([0, 1], size=trials, p=[1 - p, p])
    empirical_P_X_1 = np.mean(samples == 1)

    absolute_error = abs(empirical_P_X_1 - theoretical_P_X_1)
    return theoretical_P_X_1, theoretical_P_X_0, empirical_P_X_1, absolute_error


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    """
    STEP 1: Define Binomial(n,p) PMF:
            P(X=k) = C(n,k)p^k(1-p)^(n-k)

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 2)
            - P(X ≥ 1)

    STEP 3: Simulate 100,000 inspections
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 1)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 1)
            empirical P(X ≥ 1)

    RETURN:
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    """

    # Theoretical probabilities
    theoretical_P_0 = (1 - p) ** n
    theoretical_P_2 = math.comb(n, 2) * (p ** 2) * ((1 - p) ** (n - 2))
    theoretical_P_ge_1 = 1 - theoretical_P_0

    # Simulation
    np.random.seed(42)
    trials = 100000
    samples = np.random.binomial(n, p, size=trials)
    empirical_P_ge_1 = np.mean(samples >= 1)

    absolute_error = abs(empirical_P_ge_1 - theoretical_P_ge_1)
    return theoretical_P_0, theoretical_P_2, theoretical_P_ge_1, empirical_P_ge_1, absolute_error


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    """
    STEP 1: Let p = 1/6.

    STEP 2: Define Geometric PMF:
            P(X=k) = (5/6)^(k-1)*(1/6)

    STEP 3: Compute theoretical:
            - P(X = 1)
            - P(X = 3)
            - P(X > 4)

    STEP 4: Simulate 200,000 experiments
            using random_state=42.

    STEP 5: Compute empirical:
            - empirical P(X > 4)

    STEP 6: Compute absolute error BETWEEN:
            theoretical P(X > 4)
            empirical P(X > 4)

    RETURN:
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    """

    p = 1 / 6
    # Theoretical probabilities
    theoretical_P_1 = p
    theoretical_P_3 = ((1 - p) ** 2) * p
    theoretical_P_gt_4 = (1 - p) ** 4

    # Simulation
    np.random.seed(42)
    trials = 200000
    samples = np.random.geometric(p, size=trials)
    empirical_P_gt_4 = np.mean(samples > 4)

    absolute_error = abs(empirical_P_gt_4 - theoretical_P_gt_4)
    return theoretical_P_1, theoretical_P_3, theoretical_P_gt_4, empirical_P_gt_4, absolute_error


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    """
    STEP 1: Define Poisson PMF:
            P(X=k) = e^(-λ) λ^k / k!

    STEP 2: Compute theoretical:
            - P(X = 0)
            - P(X = 15)
            - P(X ≥ 18)

    STEP 3: Simulate 100,000 hours
            using random_state=42.

    STEP 4: Compute empirical:
            - empirical P(X ≥ 18)

    STEP 5: Compute absolute error BETWEEN:
            theoretical P(X ≥ 18)
            empirical P(X ≥ 18)

    RETURN:
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    """

    # Theoretical probabilities using iterative recurrence to avoid numerical instability
    exp_neg_lam = math.exp(-lam)
    theoretical_P_0 = exp_neg_lam
    
    # P(X=15) using recurrence: P(X=k) = P(X=k-1) * lam / k
    term = exp_neg_lam
    for _ in range(15):
        term *= lam / (_ + 1)
    theoretical_P_15 = term
    
    # CDF P(X ≤ 17) summing all probabilities using iterative method
    cdf_to_17 = 0
    term = exp_neg_lam
    for k in range(18):
        cdf_to_17 += term
        if k < 17:
            term *= lam / (k + 1)
    theoretical_P_ge_18 = 1 - cdf_to_17

    # Simulation
    np.random.seed(42)
    trials = 100000
    samples = np.random.poisson(lam, size=trials)
    empirical_P_ge_18 = np.mean(samples >= 18)

    absolute_error = abs(empirical_P_ge_18 - theoretical_P_ge_18)
    return theoretical_P_0, theoretical_P_15, theoretical_P_ge_18, empirical_P_ge_18, absolute_error
