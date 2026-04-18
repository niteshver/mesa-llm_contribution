from mesa.agent import Agent
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.model import Model
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm import Plan
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import (
    Observation,
    Reasoning,
)
from mesa_llm.tools.tool_manager import ToolManager


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The Mesa model the agent belongs to.
        reasoning (type[Reasoning]): The reasoning strategy used by the agent.
        llm_model (str): The model to use for the LLM in the format
            ``provider/model``. Defaults to ``gemini/gemini-2.0-flash``.
        system_prompt (str | None): Optional system prompt for the LLM.
        vision (float | None): Observation radius for nearby agents. Use ``-1``
            to observe all agents in the simulation.
        internal_state (list[str] | str | None): Optional internal state facts
            exposed to the reasoning module.
        step_prompt (str | None): Optional task-specific prompt used to guide
            the agent each step.
        api_base (str | None): Optional custom LiteLLM-compatible base URL for
            self-hosted or remote inference endpoints.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.
        tool_manager (ToolManager): The tool registry available to the agent.
    """

    def __init__(
        self,
        model: Model,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
        step_prompt: str | None = None,
        api_base: str | None = None,
        max_tool_retries: int = 1,
    ):
        super().__init__(model=model)

        self.model = model
        self.step_prompt = step_prompt
        self.llm = ModuleLLM(
            llm_model=llm_model, system_prompt=system_prompt, api_base=api_base
        )

        self.memory = STLTMemory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model=llm_model,
            api_base=api_base,
        )

        self.tool_manager = ToolManager()
        self.vision = vision
        self.reasoning = reasoning(agent=self)
        self.system_prompt = system_prompt
        self.max_tool_retries = max(0, max_tool_retries)
        self._current_plan = None  # Store current plan for formatting

        # display coordination
        self._step_display_data = {}

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

    async def aapply_plan(self, plan: Plan) -> list[dict]:
        """
        Asynchronous version of apply_plan.
        """
        (
            plan,
            tool_call_resp,
            retry_count,
            retry_history,
        ) = await self._aexecute_plan_with_retries(plan)

        self._current_plan = plan
        await self.memory.aadd_to_memory(
            type="action",
            content=self._build_action_memory_content(
                tool_call_resp=tool_call_resp,
                retry_count=retry_count,
                retry_history=retry_history,
            ),
        )

        return tool_call_resp

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        plan, tool_call_resp, retry_count, retry_history = (
            self._execute_plan_with_retries(plan)
        )
        self._current_plan = plan

        self.memory.add_to_memory(
            type="action",
            content=self._build_action_memory_content(
                tool_call_resp=tool_call_resp,
                retry_count=retry_count,
                retry_history=retry_history,
            ),
        )

        return tool_call_resp

    def _strip_tool_response(self, tool_call_resp: dict) -> dict:
        return {k: v for k, v in tool_call_resp.items() if k not in ["tool_call_id", "role"]}

    def _has_tool_errors(self, tool_call_resp: list[dict]) -> bool:
        return any(
            str(result.get("response", "")).startswith("Error:")
            for result in tool_call_resp
        )

    def _build_tool_retry_prompt(
        self,
        plan: Plan,
        tool_call_resp: list[dict],
        retry_attempt: int,
    ) -> str:
        original_plan = (
            str(plan.llm_plan.content).strip()
            if hasattr(plan.llm_plan, "content") and plan.llm_plan.content
            else str(plan.llm_plan).strip()
        )

        failed_results = [
            f"- {result.get('name', 'unknown_tool')}: {result.get('response', '')}"
            for result in tool_call_resp
            if str(result.get("response", "")).startswith("Error:")
        ]
        successful_results = [
            f"- {result.get('name', 'unknown_tool')}: {result.get('response', '')}"
            for result in tool_call_resp
            if not str(result.get("response", "")).startswith("Error:")
        ]

        prompt_parts = [
            f"Retry attempt {retry_attempt}: the previous tool execution failed.",
            "Issue corrected tool call(s) using the available schema only.",
            f"Original executor plan:\n{original_plan}",
            "Observed tool failures:\n" + "\n".join(failed_results),
        ]
        if successful_results:
            prompt_parts.append(
                "Previous successful tool results (avoid repeating unless needed):\n"
                + "\n".join(successful_results)
            )

        return "\n\n".join(prompt_parts)

    def _build_action_memory_content(
        self,
        *,
        tool_call_resp: list[dict],
        retry_count: int,
        retry_history: list[list[dict]],
    ) -> dict:
        content = {
            "tool_calls": [
                self._strip_tool_response(tool_call) for tool_call in tool_call_resp
            ]
        }
        if retry_count:
            content["tool_retry_count"] = retry_count
            content["tool_retry_history"] = [
                [self._strip_tool_response(tool_call) for tool_call in attempt]
                for attempt in retry_history
            ]
        return content

    def _execute_plan_with_retries(
        self, plan: Plan
    ) -> tuple[Plan, list[dict], int, list[list[dict]]]:
        current_plan = plan
        retry_history = []
        retry_count = 0

        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=current_plan.llm_plan
        )

        while retry_count < self.max_tool_retries and self._has_tool_errors(
            tool_call_resp
        ):
            retry_count += 1
            retry_history.append(tool_call_resp)
            retry_prompt = self._build_tool_retry_prompt(
                current_plan, tool_call_resp, retry_count
            )
            current_plan = self.reasoning.execute_tool_call(
                retry_prompt,
                selected_tools=current_plan.selected_tools,
                ttl=current_plan.ttl,
            )
            tool_call_resp = self.tool_manager.call_tools(
                agent=self, llm_response=current_plan.llm_plan
            )

        return current_plan, tool_call_resp, retry_count, retry_history

    async def _aexecute_plan_with_retries(
        self, plan: Plan
    ) -> tuple[Plan, list[dict], int, list[list[dict]]]:
        current_plan = plan
        retry_history = []
        retry_count = 0

        tool_call_resp = await self.tool_manager.acall_tools(
            agent=self, llm_response=current_plan.llm_plan
        )

        while retry_count < self.max_tool_retries and self._has_tool_errors(
            tool_call_resp
        ):
            retry_count += 1
            retry_history.append(tool_call_resp)
            retry_prompt = self._build_tool_retry_prompt(
                current_plan, tool_call_resp, retry_count
            )
            current_plan = await self.reasoning.aexecute_tool_call(
                retry_prompt,
                selected_tools=current_plan.selected_tools,
                ttl=current_plan.ttl,
            )
            tool_call_resp = await self.tool_manager.acall_tools(
                agent=self, llm_response=current_plan.llm_plan
            )

        return current_plan, tool_call_resp, retry_count, retry_history

    def _build_observation(self):
        """
        Construct the observation data visible to the agent at the current model step.

        This method encapsulates the shared logic used by both sync and
        async observation generation.
        This method constructs the agent's self state and determines which other
        agents are observable based on the configured vision:

        - vision > 0:
            The agent observes all agents within the specified vision radius.
        - vision == -1:
            The agent observes all agents present in the simulation.
        - vision == 0 or vision is None:
            The agent observes no other agents.

        The method supports grid-based and continuous spaces and builds a local
        state representation for all visible neighboring agents.

        Returns self_state and local_state of the agent
        """
        self_state = {
            "agent_unique_id": self.unique_id,
            "system_prompt": self.system_prompt,
            "location": (
                self.pos
                if self.pos is not None
                else (
                    getattr(self, "cell", None).coordinate
                    if getattr(self, "cell", None) is not None
                    else None
                )
            ),
            "internal_state": self.internal_state,
        }
        if self.vision is not None and self.vision > 0:
            # Check which type of space/grid the model uses
            grid = getattr(self.model, "grid", None)
            space = getattr(self.model, "space", None)

            if grid and isinstance(grid, SingleGrid | MultiGrid):
                neighbors = grid.get_neighbors(
                    tuple(self.pos),
                    moore=True,
                    include_center=False,
                    radius=self.vision,
                )
            elif grid and isinstance(
                grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid
            ):
                agent_cell = next(
                    (cell for cell in grid.all_cells if self in cell.agents),
                    None,
                )
                if agent_cell:
                    neighborhood = agent_cell.get_neighborhood(radius=self.vision)
                    neighbors = [a for cell in neighborhood for a in cell.agents]
                else:
                    neighbors = []

            elif space and isinstance(space, ContinuousSpace):
                all_nearby = space.get_neighbors(
                    self.pos, radius=self.vision, include_center=True
                )
                neighbors = [a for a in all_nearby if a is not self]

            else:
                # No recognized grid/space type
                neighbors = []

        elif self.vision == -1:
            all_agents = list(self.model.agents)
            neighbors = [agent for agent in all_agents if agent is not self]

        else:
            neighbors = []

        local_state = {}
        for i in neighbors:
            local_state[i.__class__.__name__ + " " + str(i.unique_id)] = {
                "position": (
                    i.pos
                    if i.pos is not None
                    else (
                        getattr(i, "cell", None).coordinate
                        if getattr(i, "cell", None) is not None
                        else None
                    )
                ),
                "internal_state": [
                    s for s in getattr(i, "internal_state", []) if not s.startswith("_")
                ],
            }
        return self_state, local_state

    async def agenerate_obs(self) -> Observation:
        """
        This method builds the agent's observation using the shared observation
        construction logic, stores it in the agent's memory module using
        async memory operations, and returns it as an Observation instance.
        """
        step = self.model.steps
        self_state, local_state = self._build_observation()
        await self.memory.aadd_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    def generate_obs(self) -> Observation:
        """
        This method delegates observation construction to the shared observation
        builder, stores the resulting observation in the agent's memory module,
        and returns it as an Observation instance.
        """
        step = self.model.steps
        self_state, local_state = self._build_observation()
        # Add to memory (memory handles its own display separately)
        self.memory.add_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    async def asend_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Asynchronous version of send_message.
        """
        for recipient in recipients:
            await recipient.memory.aadd_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self.unique_id,
                },
            )
        await self.memory.aadd_to_memory(
            type="message",
            content={
                "message": message,
                "sender": self.unique_id,
                "recipients": [r.unique_id for r in recipients],
            },
        )

        return f"{self} → {recipients} : {message}"

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.
        """
        for recipient in recipients:
            recipient.memory.add_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self.unique_id,
                },
            )
        self.memory.add_to_memory(
            type="message",
            content={
                "message": message,
                "sender": self.unique_id,
                "recipients": [r.unique_id for r in recipients],
            },
        )

        return f"{self} → {recipients} : {message}"

    async def apre_step(self):
        """
        Asynchronous version of pre_step.
        """
        await self.memory.aprocess_step(pre_step=True)

    async def apost_step(self):
        """
        Asynchronous version of post_step.
        """
        await self.memory.aprocess_step()

    def pre_step(self):
        """
        This is some code that is executed before the step method of the child agent is called.
        """
        self.memory.process_step(pre_step=True)

    def post_step(self):
        """
        This is some code that is executed after the step method of the child agent is called.
        It functions because of the __init_subclass__ method that creates a wrapper around the step method of the child agent.
        """
        self.memory.process_step()

    async def astep(self):
        """
        Default asynchronous step method for parallel agent execution.
        Subclasses should override this method for custom async behavior.
        If not overridden, falls back to calling the synchronous step() method.
        """
        await self.apre_step()

        if hasattr(self, "step") and self.__class__.step != LLMAgent.step:
            self.step()

        await self.apost_step()

    def __init_subclass__(cls, **kwargs):
        """
        Wrapper - allows to automatically integrate code to be executed after the step method of the child agent (created by the user) is called.
        """
        super().__init_subclass__(**kwargs)
        # only wrap if subclass actually defines its own step
        user_step = cls.__dict__.get("step")
        user_astep = cls.__dict__.get("astep")

        if user_step:

            def wrapped(self, *args, **kwargs):
                """
                This is the wrapper that is used to integrate the pre_step and post_step methods into the step method of the child agent.
                """
                LLMAgent.pre_step(self, *args, **kwargs)
                result = user_step(self, *args, **kwargs)
                LLMAgent.post_step(self, *args, **kwargs)
                return result

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args, **kwargs):
                """
                Async wrapper for astep method.
                """
                await self.apre_step()
                result = await user_astep(self, *args, **kwargs)
                await self.apost_step()
                return result

            cls.astep = awrapped
