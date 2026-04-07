# Epstein Civil Voilence

## About this Tutorial
This tutorial is inspired by official [Mesa LLM Epstein Civil Voilence Model]()

The goal of this tutorial is **not to replicate** the full original example, but to provide a simplified and beginner-friendly version. It focuses on explaining how agents interact and how the reasoning process works, helping new users understand how the model functions step by step.

#### Key Difference from original Tutorial
- A reduced no. of agent
- No custom tools
- No parallel Steeping

This makes the example easier to follow for new users while keeping library core concept.

### Model Description
This model shows how civil unrest starts and spreads. Citizens move randomly and decide whether to **protest** based on their **hardship**, fear of getting arrested, and how legitimate they think the government is. Cops try to stop protests by arresting active citizens.

### Each Citizen have 
- Inherit from `LLMAgent`
- hardship, risk aversion
- grievance
- arrest probability
Update their arrest probability by formulas. 


### Each Cop
- Also Inherit from `LLMAgent`
- Max jail term

### Tutorial Setup
Ensure you are using python 3.14 or later

## Model Execution
At each model step:
- The model advances one step
- All agents are activated using `shuffle_do("step")`.
- Each agent generate an reasoning plan and  choose an tool using `ReactReasoning` and applies it

## Creating the CitizenState
Using the preiously dependencies, we define `CitizenState` Class. Each agent have a state, the `QUIET` is the default state, State can be change if citizen can be`ACTIVE` OR `ARRESTED`.

``` python 
class CitizenState(Enum):
    QUIET = 1
    ACTIVE = 2
    ARRESTED = 3
```

## Cresting Custom Tools
In this tutorial, we use custom tool, agent excute tools as you can show in the output. First you must have to define a tool manager like `citizen_tool_manager`to `ToolManager()`. 
`@tool` is used to convert Python functions into LLM-compatible tools by automatically generating JSON schemas from type hints and `docstrings`.

#### While creating custom tools, user must remember following things:
- Always assign a variale to `Tool manager()`
- `@tool` is always used to convert python function into LLM-compatible tools.
- Always use doctrings, without it model raise an error.

``` python
citizen_tool_manager = ToolManager()
@tool(tool_manager=citizen_tool_manager)
def change_state(agent, state: str) -> str:
    """
    Change the citizen state to either QUIET or ACTIVE.

        Args:
            state: The new state for the citizen. Must be either "QUIET" or "ACTIVE".
            agent: Provided automatically.

        Returns:
            A confirmation message describing the citizen's new state.
    """
    normalized_state = state.upper()
    state_map = {
        "QUIET": CitizenState.QUIET,
        "ACTIVE": CitizenState.ACTIVE,
    }
    if normalized_state not in state_map:
        raise ValueError(f"Invalid state: {state}")

    agent.state = state_map[normalized_state]
    agent.sync_state_note()
    return f"agent {agent.unique_id} changed state to {normalized_state}."

```

## Creating Citizen Class
Each agent can calculate their arrest probability. If their arrest probability is more then threshold they can be arrested by cop. Neighbours state is `ACTIVE`. All agent can moving freely in the grid.
``` python
class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        arrest_prob_constant=0.5,                           
        regime_legitimacy=0.5,   `                          
        threshold=0.5,                                       
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.regime_legitimacy = regime_legitimacy
        self.state = CitizenState.QUIET                   # Default CitizenState
        self.vision = vision
        self.jail_sentence_left = 0  
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None                     # Update arrest_probability in function

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
        )

        self.threshold = threshold
        self.internal_state.append(
            f"tendency for risk aversion is {self.risk_aversion:.4f} on scale from 0 to 1"
        )
        self.internal_state.append(
            f"On a scale from 0 to 1, my threshold for suffering is {self.threshold:.4f}"
        )
        self.internal_state.append(
            f"On a scale of 0 to 1 my grievance due to current legitimacy of rule and personal hardships is {self.grievance:.4f}"
        )
        self.internal_state.append(
            f"tendency for risk aversion is {self.risk_aversion:.4f} on scale from 0 to 1"
        )
        self.internal_state.append(
            f"my current state in the simulation is {self.state}"
        )
        self.tool_manager = citizen_tool_manager
        self.system_prompt = "You are a citizen in a country that is experiencing civil violence. You are a member of the general population, may or may not be in active rebellion. In general, more your suffering more the tendency for you to become active. You can move one step in a nearby cell or change your state."

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself

        neighbors = self.model.grid.get_neighbors(
            tuple(self.pos), moore=True, include_center=False, radius=self.vision
        )
        for i in neighbors:
            if isinstance(i, Cop):
                cops_in_vision += 1
            elif i.state == CitizenState.ACTIVE:
                actives_in_vision += 1
    
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )
        for item in self.internal_state:
            if item.lower().startswith("my arrest probability is"):
                self.internal_state.remove(item)
                break
        self.internal_state.append(
            f"my arrest probability is {self.arrest_probability:.4f}"
        )

    def step(self):
        if self.jail_sentence_left == 0:
            self.update_estimated_arrest_probability()
            observation = self.generate_obs()
            plan = self.reasoning.plan(
                obs=observation,
                selected_tools=["change_state", "move_one_step"],
            )
            self.apply_plan(plan)
        else:
            self.jail_sentence_left -= 0.1
            if self.jail_sentence_left <= 0:
                self.jail_sentence_left = 0
                self.state = CitizenState.QUIET
                self.sync_state_note()
```
## Creating Cop class
In cop class, each agent has constant max_jail_term and when the citizen have arrest probability greater then threshold then the cop arrest citizen.

``` python
class Cop(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        max_jail_term=2,
    ):
        """
        Create a new Cop.
        Args:
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )
        self.max_jail_term = max_jail_term
        self.system_prompt = "You are a cop in a country that is experiencing civil violence. You are a member of the police force and your job is to arrest active citizens. You can arrest a citizen ONLY if they are active. You can move one step in a nearby cell or arrest a citizen."

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
        )

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        neighbors = self.model.grid.get_neighbors(
            tuple(self.pos), moore=True, include_center=False, radius=self.vision
        )
        active_citizens = [
            agent
            for agent in neighbors
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
        ]
        if active_citizens:
            citizen = self.random.choice(active_citizens)
            citizen.state = CitizenState.ARRESTED
            citizen.jail_sentence_left = self.random.randint(1, self.max_jail_term)
            citizen.sync_state_note()
            return

        observation = self.generate_obs()
        plan = self.reasoning.plan(
            obs=observation,
            selected_tools=["move_one_step"],
        )
        self.apply_plan(plan)
```
## Creating EpsteinModel Class

``` python
class EpsteinModel(Model):
    def __init__(
        self,
        initial_cops: int = 2,
        initial_citizens: int = 6,
        width: int = 10,
        height: int = 10,
        reasoning: type[Reasoning] | None = None,
        llm_model: str = "ollama/llama3.1:latest",
        vision: int = 3,
        seed=None,
    ):
        if reasoning is None:
            from mesa_llm.reasoning.react import ReActReasoning

            reasoning = ReActReasoning
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = MultiGrid(self.height, self.width, torus=False)

        model_reporters = {
            "active": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
            ),
            "quiet": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.QUIET
            ),
            "arrested": lambda m: sum(
                1
                for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ARRESTED
            ),
        }
        agent_reporters = {
            "jail_sentence": lambda a: getattr(a, "jail_sentence_left", None),
            "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
        }
        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        # ---------------------Create the cop agents---------------------
        cop_system_prompt = "You are a cop. You are tasked with arresting citizens if they are active and their arrest probability is high enough. You are also tasked with moving to a new location if there is no citizen in sight."

        agents = Cop.create_agents(
            self,
            n=initial_cops,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=cop_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Inspect your local vision and arrest a random active agent. Move if applicable.",
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_cops,))
        y = self.rng.integers(0, self.grid.height, size=(initial_cops,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

        # ---------------------Create the citizen agents---------------------
        agents = Citizen.create_agents(
            self,
            n=initial_citizens,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="",
            vision=vision,
            internal_state=None,
            step_prompt="Move around and change your state if the conditions indicate it.",
        )

        x = self.rng.integers(0, self.grid.width, size=(initial_citizens,))
        y = self.rng.integers(0, self.grid.height, size=(initial_citizens,))
        for a, i, j in zip(agents, x, y):
            self.grid.place_agent(a, (i, j))

    def step(self):
        """
        Execute one step of the model.
        """

        print(
            f"\n[bold purple] step  {self.steps} ────────────────────────────────────────────────────────────────────────────────[/bold purple]"
        )
        self.agents.shuffle_do("step")

        self.datacollector.collect(self)


```






## Important Dependencies
``` python
import math
from enum import Enum
import mesa
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager

from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from mesa_llm.reasoning.reasoning import Reasoning
```





















``` bash

#!/bin/bash

# ── Step 1 | Cop 1 ─────────────────────────────────────────────

# [Observation]

self_state:
  agent_unique_id: 1
  system_prompt: "You are a cop in a country experiencing civil violence. Your job is to arrest active citizens. You can move one step in a nearby cell or arrest a citizen."
  location: (4, 7)
  internal_state: {}

local_state:
  Cop_2:
    position: (6, 7)
    internal_state: {}

  Citizen_6:
    position: (7, 7)
    internal_state:
      risk_aversion: 0.9858
      threshold_for_suffering: 0.5000
      grievance: 0.1513
      state: QUIET

# [Plan]

reasoning: >
  Citizen 6 is currently QUIET, so cannot be arrested.
  Move closer to monitor or intercept if they become ACTIVE.
  Current position (4,7) → move North.

action: move_one_step "North"

# [Action]
tool_call:
  name: move_one_step
  result: "Agent 1 moved to (4, 8)"

# ─────────────────────────────────────────────────────────────
```










