## EV Adoption 

## Core Researcg Logic
1. Depend on 
- Financial Utility
- Social Influence   ( s = EV_neighboutr/total_neighbour)
- Infrastructure Accessinility (Like Charging sTARION)       [ I = 1/(a+ distance_to_nearest_station)]
- Environment Motivation
- Risk Perpection
- Government Policy (Optional)

## Agents
### 1. Household Agent (Main Decison)
- Income
- car type(ev or ICE)
- Env Awareness
- risk_perception  # no
- daily_milege
- home_location
- perceived_ev_cost
- utility_score
- socail_neghbour. # no
- afoption_time.   # no

### Decision Rule
```
U_EV = αF + βS + γI + δE − θR
U_ICE = baseline utility

Adop if :
U-EV > E_ICE
```

### 2. Charging Station Agent
- location
- capacity
- charginf_speed
- price_per_kwh
- queue_lenght
- utilization_rate
 

### 4. custom_Tools
#### 1. HouseHoLDAgent Tools
- Puchase_Ev
- Appy for subsidy()
- seek_social_information (Like speak_to tool used)
- Evaluate_charging_options

#### 2. ChargingsTATION Agent
- Update_capacity
- update_price

#### 3. Goverment Agent
- Invest_incharger (like make new charginf station)
- Ajust fuel prices (optinal)



