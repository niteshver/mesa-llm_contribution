🌾🚀 AgriCast-ABM
Farmer Decision & Crop Productivity Model (Mesa + Mesa-LLM)
🧠 1. Overview

AgriCast-ABM is an agent-based simulation model that captures how farmers respond to weather forecasts, social information, and uncertainty to make agricultural decisions.

The system models the full feedback loop:

Weather Forecast → Information Flow → Farmer Decisions → Crop Yield → Learning → Adoption

It integrates:

🌦 Environmental dynamics (rainfall variability)
👨‍🌾 Human decision-making (risk, trust, learning)
🌐 Social interactions (networks, diffusion)
🧠 Optional LLM-based reasoning (Mesa-LLM)
🧱 2. System Architecture
AgriCastModel
│
├── FarmerAgent (Core)
│
├── Environment
│   ├── WeatherSystem
│   ├── CropEngine
│
├── Information Layer
│   ├── Radio
│   ├── Extension Agents
│   ├── Farmer Network
│
├── Decision Engine (LLM optional)
├── Adoption Engine (LTA Model)
├── Data Collector
👨‍🌾 3. Agents
🔹 3.1 FarmerAgent (Core Entity)
Attributes
unique_id
pos

# socio-economic
land_size
wealth
education_level
risk_aversion

# trust system
trust_radio
trust_neighbors
trust_extension

# adoption
LTA                # Learning–Trust–Adoption score (0–100)
adopt_forecast     # Boolean

# farming decisions
planting_time      # early / normal / late
fertilizer
pesticide
tillage

# outputs
yield_output
profit

# memory
past_yields (list)
🔄 Behavior (Step Flow)
1. Receive forecast
2. Interact with neighbors
3. Update trust
4. Update LTA
5. Decide adoption (probabilistic)
6. Choose farming strategy
7. Compute yield
8. Update memory
9. Learn (feedback)
🔹 3.2 Information Sources
📻 Radio
Global broadcast
Accuracy-dependent influence
👨‍🔬 Extension Agents
High-trust actors
Direct farmer interaction
🌐 Farmer Network
Graph-based (social diffusion)
Peer learning
🌦️ 4. Environment
WeatherSystem
rainfall ∈ {LOW, NORMAL, HIGH}
forecast_accuracy ∈ [0.6, 0.8]
Rainfall Values
LOW    = 900 mm
NORMAL = 1300 mm
HIGH   = 1900 mm
🌱 5. Crop Engine (Simplified DSSAT)
Yield Function
yield =
base_yield
× rainfall_factor
× planting_factor
× fertilizer_factor
× pesticide_factor
× random(0.9, 1.1)
Factor Definitions
Rainfall
LOW    = 0.7
NORMAL = 1.0
HIGH   = 1.2
Planting
early + high rain  = 1.3
late + low rain    = 0.8
else               = 1.0
Fertilizer
fertilizer_factor = 1 + 0.3 * fertilizer_level
Pesticide
pesticide_factor = 1 + 0.1 * pesticide_level
🧠 6. Adoption Model (LTA)
LTA Update Equation
LTA_new =
clip(
    LTA_old
    + α * (
        trust_neighbor * info_neighbor +
        trust_network * info_network +
        trust_extension * info_extension +
        trust_radio * info_radio
    )
    + β * forecast_accuracy
    - γ * uncertainty
, 0, 100)
Feedback Learning
LTA_new = LTA_old + δ * (yield - expected_yield)
Trust Update
if forecast_correct:
    trust_radio += 0.05
else:
    trust_radio -= 0.1
🎯 7. Adoption Decision
Probability Function
P_adopt = 1 / (1 + e^(-k*(LTA - threshold)))
Final Decision
if random.random() < P_adopt:
    adopt = True
🌐 8. Interaction System
Neighbor Interaction
neighbors = grid.get_neighbors(pos, radius=r)
Network Interaction
network = nx.erdos_renyi_graph(n, p)
Information Aggregation
info = weighted average of sources
🔁 9. Simulation Loop
def step():

    # 1. Weather update
    update_weather()

    # 2. Information spread
    for farmer:
        gather_information()

    # 3. LTA update
    for farmer:
        update_LTA()

    # 4. Decision making
    for farmer:
        decide_adoption()

    # 5. Farming actions
    for farmer:
        choose_strategy()

    # 6. Crop simulation
    for farmer:
        compute_yield()

    # 7. Learning
    for farmer:
        update_memory()

    # 8. Data collection
    collect_metrics()
📊 10. Metrics
Core Metrics
average_yield
total_yield
adoption_rate
Advanced Metrics
food_security_index
information_spread_rate
trust_distribution
yield_variance
⚙️ 11. Parameters
Global Parameters
NUM_FARMERS
GRID_SIZE
FORECAST_ACCURACY
INTERACTION_RADIUS
NETWORK_PROB
Model Constants
α = 0.5
β = 0.3
γ = 0.2
δ = 0.1
k = 0.1
threshold = 50
🧠 12. Mesa-LLM Integration (Advanced)
Farmer Reasoning Prompt
You are a farmer.

Rain forecast: LOW
Neighbors suggest: plant early
Trust level: medium

Decide:
- planting time
- fertilizer usage
- whether to follow forecast
Available Tools
listen_radio
talk_to_neighbor
apply_fertilizer
plant_crop
🚀 13. Extensions
🔹 Policy Simulation
Subsidies → increase adoption
Training → increase trust
🔹 Climate Change
Increased rainfall variability
Forecast uncertainty
🔹 Economic Layer
profit = yield * price - cost
⚠️ 14. Assumptions
Single crop (maize)
Fixed land size
No market dynamics (optional)
Simplified crop model (DSSAT-inspired)
⭐ 15. Why This is GSoC-Level
✔ Hybrid system (social + environmental)
✔ Behaviorally realistic agents
✔ Learning + feedback loops
✔ Scalable architecture
i want a complete readme.md file 

Here’s a polished, complete README.md you can directly use for your repo / GSoC proposal:

🌾🚀 AgriCast-ABM
Farmer Decision & Crop Productivity Model (Mesa + Mesa-LLM)
🧠 Overview

AgriCast-ABM is an Agent-Based Model (ABM) that simulates how farmers make agricultural decisions under uncertain weather conditions.

The system models the full decision pipeline:

Weather Forecast → Information Flow → Farmer Decisions → Crop Yield → Learning → Adoption

It combines:

🌦️ Environmental simulation (weather + crops)
👨‍🌾 Human decision-making (farmers)
🌐 Social interaction (networks, extension agents)
🧠 Learning & adaptation (trust + LTA)
🤖 Optional LLM-driven reasoning (Mesa-LLM)
🧱 Architecture
AgriCastModel
│
├── FarmerAgent (core decision-maker)
│
├── Environment
│   ├── WeatherSystem
│   ├── CropEngine
│
├── Information Layer
│   ├── Radio (global broadcast)
│   ├── Extension Agents
│   ├── Farmer Network (graph-based)
│
├── Decision Engine (rule-based / LLM)
├── Adoption Engine (LTA model)
├── Data Collector
👨‍🌾 Agents
🔹 FarmerAgent (Core Entity)
Attributes
unique_id
pos

# socio-economic
land_size
wealth
education_level
risk_aversion

# trust system
trust_radio
trust_neighbors
trust_extension

# adoption
LTA                # Learning Trust Accumulation (0–100)
adopt_forecast     # Boolean

# farming decisions
planting_time      # early / normal / late
fertilizer
pesticide
tillage

# outputs
yield_output
profit

# memory
past_yields (list)
🔄 Behavior (Per Step)
1. Receive forecast
2. Interact with neighbors
3. Update trust
4. Update LTA
5. Decide adoption (probabilistic)
6. Choose farming strategy
7. Compute yield
8. Update memory
9. Learn (feedback)
📡 Information Sources
📻 Radio
Broadcasts weather forecast globally
Accuracy depends on system parameter
👨‍🔬 Extension Agents
High-trust advisory entities
Direct influence on farmers
🌐 Farmer Network
Graph-based (e.g., Erdős–Rényi)
Enables peer-to-peer information diffusion
🌦️ Environment
WeatherSystem
rainfall ∈ {LOW, NORMAL, HIGH}
forecast_accuracy ∈ [0.6, 0.8]
Rainfall Values
LOW    = 900 mm
NORMAL = 1300 mm
HIGH   = 1900 mm
🌱 Crop Engine (Simplified DSSAT)
Yield Function
yield =
base_yield
× rainfall_factor
× planting_factor
× fertilizer_factor
× pesticide_factor
× random(0.9, 1.1)
Factors
🌧️ Rainfall
LOW    = 0.7
NORMAL = 1.0
HIGH   = 1.2
🌱 Planting Strategy
early + high rain  = 1.3
late + low rain    = 0.8
else               = 1.0
🧪 Fertilizer
fertilizer_factor = 1 + 0.3 * fertilizer_level
🐛 Pesticide
pesticide_factor = 1 + 0.1 * pesticide_level
🧠 Adoption Model (LTA)
🔹 LTA Update
LTA_new =
clip(
    LTA_old
    + α * (
        trust_neighbor * info_neighbor +
        trust_network * info_network +
        trust_extension * info_extension +
        trust_radio * info_radio
    )
    + β * forecast_accuracy
    - γ * uncertainty
, 0, 100)
🔹 Feedback Learning
LTA_new = LTA_old + δ * (yield - expected_yield)
🔹 Trust Update
if forecast_correct:
    trust_radio += 0.05
else:
    trust_radio -= 0.1
🎯 Adoption Decision
Probability Function
P_adopt = 1 / (1 + e^(-k*(LTA - threshold)))
Final Decision
if random.random() < P_adopt:
    adopt = True
🌐 Interaction System
Neighbor Interaction
neighbors = grid.get_neighbors(pos, radius=r)
Network Interaction
network = nx.erdos_renyi_graph(n, p)
Information Aggregation
info = weighted average of all sources
🔁 Simulation Loop
def step():

    # 1. Weather update
    update_weather()

    # 2. Information spread
    for farmer:
        gather_information()

    # 3. LTA update
    for farmer:
        update_LTA()

    # 4. Decision
    for farmer:
        decide_adoption()

    # 5. Farming actions
    for farmer:
        choose_strategy()

    # 6. Crop simulation
    for farmer:
        compute_yield()

    # 7. Learning
    for farmer:
        update_memory()

    # 8. Data collection
    collect_metrics()
📊 Metrics
Core Metrics
average_yield
total_yield
adoption_rate
Advanced Metrics
food_security_index
information_spread_rate
trust_distribution
yield_variance
⚙️ Parameters
Global Parameters
NUM_FARMERS
GRID_SIZE
FORECAST_ACCURACY
INTERACTION_RADIUS
NETWORK_PROB
Model Constants
α = 0.5
β = 0.3
γ = 0.2
δ = 0.1
k = 0.1
threshold = 50
🤖 Mesa-LLM Extension (Advanced)
Farmer Reasoning Prompt
You are a farmer.

Rain forecast: LOW  
Neighbors suggest: plant early  
Trust level: medium  

Decide:
- planting time
- fertilizer usage
- whether to follow forecast
Tools
listen_radio
talk_to_neighbor
apply_fertilizer
plant_crop
🚀 Extensions
🏛️ Policy Simulation
subsidy → increases adoption  
training → increases trust  
🌍 Climate Change
increase rainfall variability
💰 Economic Layer
profit = yield * price - cost
⚠️ Assumptions
- Single crop (maize)
- Fixed land size
- No market dynamics (optional)
- Simplified crop model
⭐ Why This is GSoC-Level
✔ Hybrid system (social + environmental)
✔ Incorporates learning & feedback loops
✔ Realistic farmer behavior modeling
✔ Scalable & modular architecture
✔ Supports both rule-based and LLM-driven agents
🏁 Summary
Farmers receive information
→ interact socially
→ update beliefs (LTA)
→ make decisions
→ affect yield
→ learn from outcomes
→ system evolves
🚀 Next Steps

You can extend this project by:

Implementing Mesa simulation (model.py, agent.py)
Adding LLM agents (Mesa-LLM)
Building a UI with Solara
Running policy experiments
Publishing results

If you want, I can now generate:

👉 Full working code (agent.py, model.py, app.py)
👉 Or a GSoC proposal draft based on this README

We use cookies

We use cookies to help this site function, understand service usage, and support marketing efforts. Visit Manage Cookies to change preferences anytime. View our Cookie Policy for more info.