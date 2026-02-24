# Epstein Civil Violence Model Tutorial (mesa-llm)

## About This Tutorial

This tutorial is inspired by the official
[Mesa-llm Epstein Civil Violence Example](https://github.com/mesa/mesa-llm/tree/main/examples/epstein_civil_violence)

The goal here is to present a **simplified and tutorial-friendly version** that focuses on how LLM-powered agents interact in a spatial grid environment with emergent collective behavior.

Key differences from the full example include:
- Reduced number of agents for clarity
- Simplified visualization setup
- Emphasis on understanding tool-based reasoning and spatial dynamics
- Focus on the core rebellion mechanics

This makes the example easier to follow for new users while still demonstrating how mesa-llm enables complex agent-based models with language-based reasoning.

## Model Description

This model simulates how civil unrest grows and is suppressed, based on Joshua Epstein's classic agent-based model. The simulation includes:

### Citizen Agents
- Wander the grid randomly
- Each has individual `hardship`, `risk_aversion`, and `threshold` values
- Share a common perception of `regime_legitimacy`
- Decide whether to rebel based on grievance and perceived arrest probability

### Cop Agents
- Patrol the grid
- Arrest active (rebelling) Citizens within their vision
- Work on behalf of the regime

### Key Dynamics
The model generates mass uprisings as self-reinforcing processes:
- If enough agents are rebelling, the probability of any individual being arrested decreases
- This makes more agents likely to join the uprising
- However, as Cops arrest rebelling Citizens, fewer agents become willing to join

Unlike traditional Mesa implementations, this version uses **LLM-powered reasoning** where agents think and decide using natural language reasoning and tools.

## Tutorial Setup

Ensure you are using Python 3.12 or later.

### Install mesa-llm and required packages

```bash
pip install -U mesa-llm
```

## Model Mechanics

### Rebellion Rule
A Citizen becomes (or remains) **active** (i.e., rebels) if:

```
grievance - (risk_aversion * arrest_probability) > threshold
```

Where:

```
grievance = hardship * (1 - regime_legitimacy)
```

### Arrest Probability
The perceived probability of arrest is calculated as:

```
arrest_probability = 1 - exp(-k * round(cops_in_vision / actives_in_vision))
```

Where:
- `k` is a constant (default 0.5)
- `cops_in_vision` is the number of cops within the agent's vision
- `actives_in_vision` is the number of active Citizens (including the agent itself)

## Why This Model Uses a Spatial Grid

Unlike the negotiation tutorial, this model requires a **spatial grid** because:
- Agents have limited **vision** and can only observe nearby agents
- Cops can only arrest Citizens within their vision radius
- The probability of arrest depends on the **local** ratio of cops to active citizens
- Movement and spatial positioning directly affect agent decisions

The spatial dynamics are essential to the model's emergent behavior and cannot be simplified away.

## Model Execution

At each model step:
1. The model advances one step
2. All agents are activated using `shuffle_do("step")`
3. Each Citizen:
   - Updates their estimated arrest probability based on local observations
   - Uses reasoning to decide whether to change state or move
4. Each Cop:
   - Uses reasoning to decide whether to arrest an active Citizen or move

## Agent Tools

Agents in this model use **tools** to take actions:

### Citizen Tools
- `change_state`: Change between "QUIET" and "ACTIVE" states
- `move_one_step`: Move to a neighboring cell on the grid

### Cop Tools
- `arrest_citizen`: Arrest an active Citizen by their ID
- `move_one_step`: Move to a neighboring cell on the grid

These tools are made available to the LLM during reasoning, allowing agents to take concrete actions in the simulation.

## Creating the Citizen Class

Citizens are the primary agents in this model. They decide whether to rebel based on their grievance, risk aversion, and the perceived probability of arrest.

```python
# Import Dependencies
from mesa.model import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager
from mesa_llm.tools.tool_decorator import tool
import mesa.discrete_space
import math
from enum import Enum

# Define citizen states
class CitizenState(Enum):
    QUIET = 1
    ACTIVE = 2
    ARRESTED = 3

# Create tool managers
citizen_tool_manager = ToolManager()
cop_tool_manager = ToolManager()

# ---------------- CITIZEN ----------------
class Citizen(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A member of the general population, may or may not be in active rebellion.

    The citizen decides whether to rebel based on:
    - Personal hardship
    - Perceived regime legitimacy
    - Risk aversion
    - Estimated probability of arrest
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
        regime_legitimacy=0.5,
        threshold=0.5,
    ):
        # Initialize the LLM agent with grid capabilities
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        # Set individual characteristics (drawn randomly)
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()

        # Set shared and constant values
        self.regime_legitimacy = regime_legitimacy
        self.state = CitizenState.QUIET
        self.vision = vision
        self.jail_sentence_left = 0
        self.arrest_prob_constant = arrest_prob_constant
        self.threshold = threshold

        # Calculate grievance based on hardship and legitimacy
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_probability = None

        # Attach memory to maintain context across steps
        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",
        )

        # Set up the agent's internal state for reasoning
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
            f"my current state in the simulation is {self.state}"
        )

        # Attach tool manager
        self.tool_manager = citizen_tool_manager

        # Set system prompt
        self.system_prompt = """You are a citizen in a country experiencing civil violence.
        You are a member of the general population, may or may not be in active rebellion.
        In general, the more you suffer, the more tendency you have to become active.
        You can move one step to a nearby cell or change your state."""

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood,
        estimate the probability of arrest if I go active.
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself

        # Get neighbors within vision radius
        neighbors = self.model.grid.get_neighbors(
            tuple(self.pos),
            moore=True,
            include_center=False,
            radius=self.vision
        )

        # Count cops and active citizens
        for neighbor in neighbors:
            if isinstance(neighbor, Cop):
                cops_in_vision += 1
            elif hasattr(neighbor, 'state') and neighbor.state == CitizenState.ACTIVE:
                actives_in_vision += 1

        # Calculate arrest probability
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )

        # Update internal state with new arrest probability
        for item in self.internal_state:
            if item.lower().startswith("my arrest probability is"):
                self.internal_state.remove(item)
                break
        self.internal_state.append(
            f"my arrest probability is {self.arrest_probability:.4f}"
        )

    def step(self):
        """
        Called once per model step.

        The citizen:
        1. Updates estimated arrest probability based on local vision
        2. Generates an observation including their current state
        3. Uses reasoning to decide whether to change state or move
        """
        # Only act if not in jail
        if self.jail_sentence_left == 0:
            # Update arrest probability based on local environment
            self.update_estimated_arrest_probability()

            # Generate observation for reasoning module
            observation = self.generate_obs()

            # Use reasoning to generate a plan
            plan = self.reasoning.plan(
                obs=observation,
                selected_tools=["change_state", "move_one_step"],
            )

            # Apply the plan
            self.apply_plan(plan)
        else:
            # Serve jail sentence
            self.jail_sentence_left -= 0.1
```

## Creating Citizen Tools

Citizens need tools to change their state and move on the grid:

```python
# ---------------- CITIZEN TOOLS ----------------
@tool(tool_manager=citizen_tool_manager)
def change_state(agent: "LLMAgent", state: str) -> str:
    """
    Change the state of the citizen agent.

    Args:
        state: The state to change to. Must be "QUIET" or "ACTIVE"
        agent: Provided automatically

    Returns:
        A string confirming the agent's new state
    """
    state_map = {
        "QUIET": CitizenState.QUIET,
        "ACTIVE": CitizenState.ACTIVE,
    }

    if state not in state_map:
        raise ValueError(f"Invalid state: {state}")

    agent.state = state_map[state]
    return f"agent {agent.unique_id} changed state to {state}."

# Note: move_one_step tool is provided by mesa_llm.tools.inbuilt_tools
# and is automatically available to agents on a grid
```

## Creating the Cop Class

Cops patrol the grid and arrest active Citizens:

```python
# ---------------- COP ----------------
class Cop(LLMAgent, mesa.discrete_space.CellAgent):
    """
    A cop agent that patrols and arrests active citizens.

    The cop inspects its local vision and arrests active citizens
    or moves to a new location.
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
        # Initialize the LLM agent with grid capabilities
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
        self.tool_manager = cop_tool_manager

        # Set system prompt
        self.system_prompt = """You are a cop in a country experiencing civil violence.
        You are a member of the police force and your job is to arrest active citizens.
        You can arrest a citizen ONLY if they are active.
        You can move one step to a nearby cell or arrest a citizen."""

        # Attach memory
        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",
        )

    def step(self):
        """
        Called once per model step.

        The cop:
        1. Generates an observation of their local environment
        2. Uses reasoning to decide whether to arrest or move
        """
        # Generate observation for reasoning module
        observation = self.generate_obs()

        # Use reasoning to generate a plan
        plan = self.reasoning.plan(
            obs=observation,
            selected_tools=["move_one_step", "arrest_citizen"],
        )

        # Apply the plan
        self.apply_plan(plan)
```

## Creating Cop Tools

Cops need a tool to arrest citizens:

```python
# ---------------- COP TOOLS ----------------
@tool(tool_manager=cop_tool_manager)
def arrest_citizen(agent: "LLMAgent", citizen_id: int) -> str:
    """
    Arrest a citizen (only if they are active).

    Args:
        citizen_id: The unique id of the citizen to arrest
        agent: Provided automatically

    Returns:
        A string confirming the citizen's arrest
    """
    import random

    # Find the citizen by ID
    citizen = next(
        (a for a in agent.model.agents if a.unique_id == citizen_id),
        None
    )

    if citizen is None:
        return f"Citizen {citizen_id} not found."

    # Arrest the citizen
    citizen.state = CitizenState.ARRESTED
    citizen.jail_sentence_left = random.randint(1, agent.max_jail_term)

    return f"Citizen {citizen_id} arrested by cop {agent.unique_id}."
```

## Creating the Epstein Civil Violence Model

The model sets up the grid environment, creates agents, and coordinates their actions:

```python
# ---------------- MODEL ----------------
class EpsteinModel(Model):
    """
    Model coordinating the Epstein civil violence simulation.

    The model:
    - Creates a spatial grid
    - Populates it with Citizens and Cops
    - Advances the simulation step by step
    - Collects data on agent states
    """

    def __init__(
        self,
        initial_cops: int = 5,
        initial_citizens: int = 15,
        width: int = 10,
        height: int = 10,
        reasoning: type = ReActReasoning,
        llm_model: str = "ollama/llama3",
        vision: int = 2,
        seed=None,
    ):
        # Initialize the Mesa model
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.vision = vision

        # Create a multi-grid (multiple agents per cell allowed)
        self.grid = MultiGrid(self.width, self.height, torus=False)

        # Set up data collection
        model_reporters = {
            "active": lambda m: sum(
                1 for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE
            ),
            "quiet": lambda m: sum(
                1 for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.QUIET
            ),
            "arrested": lambda m: sum(
                1 for agent in m.agents
                if isinstance(agent, Citizen) and agent.state == CitizenState.ARRESTED
            ),
        }

        self.datacollector = DataCollector(model_reporters=model_reporters)

        # ---------------------Create cop agents---------------------
        cop_system_prompt = """You are a cop tasked with arresting citizens if they are active.
        You are also tasked with moving to a new location if there is no citizen in sight."""

        cops = Cop.create_agents(
            self,
            n=initial_cops,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=cop_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt="Inspect your local vision and arrest a random active agent. Move if applicable.",
        )

        # Place cops randomly on the grid
        x = self.rng.integers(0, self.grid.width, size=(initial_cops,))
        y = self.rng.integers(0, self.grid.height, size=(initial_cops,))
        for cop, i, j in zip(cops, x, y):
            self.grid.place_agent(cop, (i, j))

        # ---------------------Create citizen agents---------------------
        citizens = Citizen.create_agents(
            self,
            n=initial_citizens,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt="",
            vision=vision,
            internal_state=None,
            step_prompt="Move around and change your state if the conditions indicate it.",
        )

        # Place citizens randomly on the grid
        x = self.rng.integers(0, self.grid.width, size=(initial_citizens,))
        y = self.rng.integers(0, self.grid.height, size=(initial_citizens,))
        for citizen, i, j in zip(citizens, x, y):
            self.grid.place_agent(citizen, (i, j))

    def step(self):
        """
        Advance the model by one step.

        At each step:
        - All agents are activated once in random order
        - Each agent performs its reasoning and actions
        - Data is collected for analysis
        """
        print(f"\n--- Model step {self.steps} ---")

        # Activate all agents in random order
        self.agents.shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)
```

## Running the Model

Create an instance of the model and run it for several steps:

```python
# ---------------- RUN ----------------
if __name__ == "__main__":
    # Create an instance of the Epstein civil violence model
    model = EpsteinModel(
        initial_cops=5,
        initial_citizens=15,
        width=10,
        height=10,
        llm_model="ollama/llama3",
        vision=2,
    )

    # Run the model for a fixed number of steps
    for _ in range(5):
        model.step()

    # Get collected data
    data = model.datacollector.get_model_vars_dataframe()
    print("\n--- Simulation Results ---")
    print(data)
```

## Understanding the Output

The model will print reasoning traces for each agent at each step, showing how they decide to:
- Change state (Citizens becoming active or quiet)
- Move to new positions
- Arrest citizens (Cops)

You'll also see the emergent dynamics:
- Initial quiet period as Citizens assess their environment
- Potential uprising as some Citizens become active
- Cops responding to arrest active Citizens
- The system potentially stabilizing or continuing to fluctuate

## Observing Emergent Behavior

The key insight of this model is the **self-reinforcing nature** of civil unrest:

1. When few agents are active, arrest probability is high, deterring rebellion
2. As more agents become active, arrest probability for each individual decreases
3. This encourages more agents to join, creating a cascade effect
4. Cops arresting active Citizens can reverse this trend

This emergence happens **without any coordination** between Citizens—it's purely the result of individual agents reasoning about their local environment.

## Exercises

Try these exercises to deepen your understanding:

1. **Vary the number of Cops**
   - Increase or decrease `initial_cops` and observe how it affects uprising dynamics
   - What ratio of Cops to Citizens seems to maintain stability?

2. **Adjust vision radius**
   - Change the `vision` parameter for agents
   - How does limited vision affect the model's behavior?

3. **Modify regime legitimacy**
   - Add a `regime_legitimacy` parameter to the model initialization
   - Lower values should increase grievance and rebellion likelihood

4. **Track spatial patterns**
   - Use the datacollector to track where uprisings occur on the grid
   - Do they cluster or spread randomly?

5. **Change reasoning strategies**
   - Try different reasoning modules (e.g., `CoTReasoning` instead of `ReActReasoning`)
   - How does this affect agent decisions?

## Key Takeaways

This tutorial demonstrates:
- How to create **spatial agent-based models** with mesa-llm
- Using **tools** to enable LLM agents to take concrete actions
- Implementing **dual agent types** (Citizens and Cops) with different behaviors
- How **local observations** and **limited vision** create emergent global patterns
- The power of **language-based reasoning** for complex decision-making in ABMs

The Epstein civil violence model shows that LLM-powered agents can replicate and extend classic agent-based modeling scenarios while adding interpretable, flexible reasoning capabilities.
