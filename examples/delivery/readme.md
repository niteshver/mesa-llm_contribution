# Simple Delivery Agent Model

**Disclaimer**: This is a toy model designed for illustrative purposes. It is not intended to represent a real-world logistics or delivery optimization system.

## Summary

This model simulates a **quick-commerce style delivery scenario** (inspired by platforms like Zepto or Swiggy Instamart), where a delivery agent must deliver an order to a customer within a strict time limit (e.g., 10 minutes).

The objective of the model is to study how **distance and environmental delays** (such as traffic, rain, or road issues) affect delivery success, while demonstrating clean usage of **Mesa** and **mesa-llm** concepts such as agents, tools, reasoning separation, and data collection.

## Model Concept

- A **delivery agent** starts at a location in a continuous 2D space.
- A **customer** is placed at another location.
- The agent calculates the distance to the customer and estimates delivery time.
- Random environmental delays are applied.
- If the delivery is completed within the allowed time, it succeeds; otherwise, it fails.

The model focuses on **clarity, explainability, and correctness**, rather than realism or optimization.


## Agents

### Delivery Agent

The delivery agent represents a rider responsible for fulfilling an order.

**State variables include**:
- Position `(x, y)`
- Speed
- Maximum allowed delivery time (e.g., 10 minutes)
- Time spent on delivery
- Earnings
- Delivery status (idle, delivered, failed)

### Customer

The customer is a passive agent with:
- A fixed location `(x, y)`
- An active delivery request

---

## Agent Decision Logic

The delivery agent follows a **fully deterministic rule-based process**:

1. Compute the distance to the customer.
2. Calculate base travel time using agent speed.
3. Apply environmental delays (traffic, rain, road issues).
4. Compare total delivery time with the maximum allowed time.
5. Deliver the order if within time; otherwise, mark it as failed.

The LLM does **not** make decisions. It is used only to explain outcomes in natural language.


## Mathematical Formulation

### Distance

Euclidean distance between agent and customer:

\[
d = \sqrt{(x_a - x_c)^2 + (y_a - y_c)^2}
\]

### Base Travel Time

\[
t_{base} = \frac{d}{v}
\]

where `v` is the agent’s speed.

### Environmental Delay

Delays are additive:

- Traffic delay
- Rain delay
- Road condition delay

\[
t_{delay} = t_{traffic} + t_{rain} + t_{road}
\]

### Total Delivery Time

\[
t_{total} = t_{base} + t_{delay}
\]

### Delivery Outcome

- Successful if `t_total ≤ max_time`
- Failed otherwise

---

## Tools

The model uses a **minimal and explicit tool set** to mutate the environment:

- `calculate_delivery_time`: Computes delivery duration using distance and delays.
- `deliver_product`: Marks delivery completion, updates earnings and agent state.
- Optional delay tools introduce bounded randomness (traffic, rain, road).

Tools are intentionally simple to avoid hallucinations and keep behavior explainable.

---

## Data Collection

The model tracks:
- Total deliveries attempted
- Successful vs failed deliveries
- Average delivery time
- Total agent earnings

This data can be plotted to analyze delivery efficiency.

---

## Visualization

The visualization shows:
- Delivery agent position
- Customer location
- Time-series plots of delivery success vs failure

The model uses a continuous space to keep movement and visualization intuitive.


## Complex Model
This is like simple agent model but we add a proper graph, grid , location of delivery agent and custoer , we find shortest path using djjkastra algorith, and when agent deliver the product they have limited time like 10 minutes , and in the path ther are traffic,road,and agent earn money , tools like deliver the product are used, and ther are multiple delivery agent and multiple customer so its like real but it is comples and 