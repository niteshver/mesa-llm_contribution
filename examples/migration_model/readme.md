# 🧠 Conflict-Driven Migration Agent-Based Model (ABM)

## 📌 Overall Structure

### Agent Types

-   **Person agents** (individual individuals)\
-   **Household agents** (decision-making unit)

------------------------------------------------------------------------

## 🔄 Daily Migration Process

Migration occurs in **7 stages** each day:

1.  Conflict impact calculation\
2.  Attitude formation\
3.  Perceived behavior control\
4.  Risk-to-probability conversion\
5.  Household aggregation\
6.  Bernoulli sampling\
7.  Peer threshold adjustment

------------------------------------------------------------------------

# 🔴 STEP 1 --- Conflict Impact on Person

For each conflict event j affecting person i:

## Event Impact Formula

Impact_i,j(t) = I_j / ((1 + δ·d(i,j)) · (1 + τ·Δt))

Where:

-   I_j = intensity of event\
-   d(i,j) = spatial distance between agent and event\
-   Δt = time difference\
-   δ = spatial decay parameter\
-   τ = temporal decay parameter

👉 Closer and recent events have stronger impact.

------------------------------------------------------------------------

# 🔴 STEP 2 --- Attitude Toward Risk

Total accumulated risk from all past events:

A_i(t) = Σ\_{j ∈ E_t} Impact_i,j(t)

Where:

-   E_t = all past conflict events until time t

This corresponds to **Attitude** in the Theory of Planned Behavior.

------------------------------------------------------------------------

# 🔴 STEP 3 --- Perceived Behavior Control (PBC)

P_i(t) = α_i · A_i(t) + θ · P_i(t−1)

Where:

-   α_i = risk-proneness (age, gender, etc.)\
-   θ = memory retention parameter\
-   P_i(t−1) = previous perceived risk

This introduces memory effects and demographic heterogeneity.

------------------------------------------------------------------------

# 🔴 STEP 4 --- Convert Risk to Migration Probability

Pr_i(t) = 1 / (1 + e\^(−v(P_i(t) − Q)))

Where:

-   v = growth rate (risk sensitivity)\
-   Q = baseline migration control

Output range: 0 to 1

------------------------------------------------------------------------

# 🔴 STEP 5 --- Household Aggregation

Pr_H(t) = (1 / \|H\|) Σ\_{i ∈ H} Pr_i(t)

Where:

-   \|H\| = number of household members

------------------------------------------------------------------------

# 🔴 STEP 6 --- Bernoulli Sampling

M_H(t) \~ Bernoulli(Pr_H(t))

If result = 1 → household migrates\
All members migrate together.

------------------------------------------------------------------------

# 🔴 STEP 7 --- Inter-Household Peer Effect (Threshold Model)

Let:

-   N_H = neighboring households\
-   φ = threshold parameter

If:

(Migrated Neighbors / \|N_H\|) \> φ

Then:

Pr_H(t) = 1

This models herd behavior (Granovetter threshold model).

------------------------------------------------------------------------

# 📊 Final Daily Output

Each day, the model records:

-   Total refugees\
-   Refugees by age\
-   Refugees by gender\
-   Refugees by region

------------------------------------------------------------------------

# ⚙️ Model Parameters

  Parameter   Meaning
  ----------- ----------------------
  δ           Spatial decay
  τ           Temporal decay
  θ           Memory decay
  v           Logistic growth rate
  Q           Baseline migration
  φ           Peer threshold

Parameters are calibrated using real border-crossing data.

------------------------------------------------------------------------

# 🏗 Complete Migration Flow (Pseudocode)

    for each day:

        for each person:
            compute distance-based conflict impact
            compute total accumulated risk
            apply demographic sensitivity
            convert to migration probability

        for each household:
            average member probabilities
            apply Bernoulli sampling
            check neighbor threshold
            migrate if triggered

        remove migrated households
        proceed to next day
