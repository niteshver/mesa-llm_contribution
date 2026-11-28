"""Snapshot-staged archetype reasoning for Mesa-LLM.

At each model step, call manager.prepare_step(agents) BEFORE any agent acts,
then agent.step() (each agent should use plan/apply_plan), then finish_step().
New archetype membership takes effect at the next step. MVP supports only
mutually exclusive, argument-free, registered actions; not tool chains.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from mesa_llm.actions.action_manager import ActionChoice
from mesa_llm.reasoning.reasoning import _UNSET, Observation, Plan, Reasoning, ToolSelection


@dataclass
class ArchetypePlan(Plan):
    action: ActionChoice | None = None
    probabilities: dict[str, float] | None = None
    archetype_key: Hashable | None = None


class ArchetypeManager:
    """A model-scoped estimator; never instantiate separately per agent."""

    def __init__(self, n: int):
        self.n = n
        self.cache: dict[tuple, dict[str, float]] = {}
        self.inflight: dict[tuple, asyncio.Task] = {}
        self.step: int | None = None
        self.llm_calls = 0
        self.cache_hits = 0
        self._prepared_step: int | None = None
        self._plans: dict[Any, ArchetypePlan] = {}
        self._prepared_agents: set[Any] = set()

    def prepare(self, step: int):
        if self.step != step:
            self.cache.clear()
            self.step = step

    def prepare_step(self, agents):
        """Freeze observations/keys and sample ALL decisions before execution.

        Only agents using this manager are included. Call this from Model.step
        before invoking any agent.step. This method does NOT execute actions.
        """
        agents = list(agents)
        if not agents:
            self._plans = {}
            self._prepared_agents = set()
            self._prepared_step = None
            return
        model = agents[0].model
        step = model.steps
        if self._prepared_step is not None:
            raise RuntimeError("Already prepared a timestep; call finish_step first")
        entries = []
        for agent in agents:
            if agent.model is not model:
                raise ValueError("All prepared agents must belong to the same model")
            reasoning = getattr(agent, "reasoning", None)
            if not isinstance(reasoning, ArchetypeReasoning) or reasoning.manager is not self:
                raise ValueError("All prepared agents must use the same archetype configuration")
            obs = agent.generate_obs()
            key, context, names = reasoning._inputs(obs, None)
            entries.append((agent, reasoning, obs, key, context, names))

        # All observations and keys have been captured before any model mutation.
        # Stage the result locally so an LLM error never publishes partial plans.
        plans = {}
        for agent, reasoning, obs, key, context, names in entries:
            probabilities = self.estimate(
                agent=agent, step=obs.step, archetype=key,
                context=context, names=names,
            )
            plans[agent] = reasoning._sample(obs, key, probabilities, 1)
        self._plans = plans
        self._prepared_agents = set(plans)
        self._prepared_step = step

    def get_plan(self, agent):
        """Return the frozen per-agent plan; never recompute mid-timestep."""
        if self._prepared_step != agent.model.steps or agent not in self._plans:
            raise RuntimeError(
                "Archetype step is not prepared. In Model.step, call "
                "agent.reasoning.manager.prepare_step(agents) before agent steps."
            )
        return self._plans[agent]

    def finish_step(self):
        """Discard frozen decisions; next Model.step recomputes membership."""
        self._plans = {}
        self._prepared_agents = set()
        self._prepared_step = None

    @staticmethod
    def decode(response, names: tuple[str, ...]) -> str:
        try:
            value = response.choices[0].message.content.strip()
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Expected an LLM response with text content") from exc
        if value.startswith("{"):
            try:
                value = json.loads(value)["action"]
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError("Expected JSON containing 'action'") from exc
        for name in names:
            if isinstance(value, str) and value.casefold() == name.casefold():
                return name
        raise ValueError(f"Unexpected action {value!r}; choose one of {names}")

    @staticmethod
    def make_prompt(context: str, names: tuple[str, ...]) -> str:
        return (f"{context}\n\nChoose exactly ONE action from: {', '.join(names)}. "
                "Reply ONLY with the exact action name.")

    def cache_key(self, agent, step, archetype, context, names):
        llm = agent.llm
        return (step, archetype, context, names,
                getattr(llm, "active_model", getattr(llm, "llm_model", "")),
                str(agent.system_prompt or ""))

    def estimate(self, *, agent, step, archetype, context, names):
        self.prepare(step)
        key = self.cache_key(agent, step, archetype, context, names)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key].copy()
        prompt = self.make_prompt(context, names)
        outputs = []
        for _ in range(self.n):
            self.llm_calls += 1
            response = agent.llm.generate(prompt=prompt, tool_schema=None,
                                          tool_choice="none",
                                          system_prompt=agent.system_prompt,
                                          suppress_thinking=True)
            outputs.append(self.decode(response, names))
        counts = Counter(outputs)
        distribution = {name: counts[name] / self.n for name in names}
        self.cache[key] = distribution
        return distribution.copy()

    async def aestimate(self, *, agent, step, archetype, context, names):
        self.prepare(step)
        key = self.cache_key(agent, step, archetype, context, names)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key].copy()
        # Include loop identity to keep asyncio Tasks scoped to their event loop.
        async_key = (asyncio.get_running_loop(), *key)
        task = self.inflight.get(async_key)
        if task is None:
            async def calculate():
                prompt = self.make_prompt(context, names)
                outputs = []
                for _ in range(self.n):
                    self.llm_calls += 1
                    response = await agent.llm.agenerate(
                        prompt=prompt, tool_schema=None, tool_choice="none",
                        system_prompt=agent.system_prompt, suppress_thinking=True)
                    outputs.append(self.decode(response, names))
                counts = Counter(outputs)
                distribution = {name: counts[name] / self.n for name in names}
                if self.step == step:
                    self.cache[key] = distribution
                return distribution
            task = asyncio.create_task(calculate())
            self.inflight[async_key] = task
            task.add_done_callback(lambda _task, k=async_key: self.inflight.pop(k, None))
        else:
            self.cache_hits += 1
        return (await asyncio.shield(task)).copy()


class ArchetypeReasoning(Reasoning):
    """Per-agent adapter using model-shared probability estimates.

    ``configure(...)`` returns a new Reasoning *class*, preserving Mesa-LLM's
    existing ``reasoning(agent=self)`` agent initialization convention.
    """

    _n = 3
    _key_fn: Callable | None = None
    _prompt_fn: Callable | None = None

    @classmethod
    def configure(cls, *, n: int = 3, key_fn: Callable, prompt_fn: Callable | None = None):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer (LLM samples)")
        if not callable(key_fn) or (prompt_fn is not None and not callable(prompt_fn)):
            raise TypeError("key_fn and prompt_fn must be callable")
        return type(f"Configured{cls.__name__}", (cls,), {
            "_n": n,
            "_key_fn": staticmethod(key_fn),
            "_prompt_fn": staticmethod(prompt_fn) if prompt_fn else None,
        })

    @property
    def manager(self) -> ArchetypeManager:
        model = self.agent.model
        managers = getattr(model, "_mesa_llm_archetype_managers", None)
        if managers is None:
            managers = {}
            model._mesa_llm_archetype_managers = managers
        cls = type(self)
        if cls not in managers:
            managers[cls] = ArchetypeManager(self._n)
        return managers[cls]

    def _inputs(self, obs: Observation, prompt: str | None):
        if self._key_fn is None:
            raise ValueError("Pass ArchetypeReasoning.configure(key_fn=..., n=...) as reasoning")
        key = self._key_fn(self.agent, obs)
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError("key_fn must return a hashable value") from exc
        actions = self.agent._action_manager.available_actions(agent=self.agent)
        names = tuple(sorted(actions))
        if not names:
            raise ValueError("ArchetypeReasoning requires registered agent actions")
        for name in names:
            try:
                self.agent._action_manager.validate(self.agent, ActionChoice(name=name, arguments={}))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Action {name!r} must accept no user arguments") from exc
        context = self._prompt_fn(key) if self._prompt_fn else f"Archetype: {key!r}."
        if not isinstance(context, str) or not context.strip():
            raise ValueError("prompt_fn must return a non-empty string")
        decision = prompt if prompt is not None else self.agent.step_prompt
        if decision:
            context += f"\nDecision: {decision}"
        return key, context, names

    def _sample(self, obs, key, probabilities, ttl):
        rng = self.agent.model.random  # Mesa model RNG; seeded by model
        r = rng.random()
        total = 0.0
        name = next(reversed(probabilities))
        for candidate, probability in probabilities.items():
            total += probability
            if r < total:
                name = candidate
                break
        choice = self.agent._action_manager.validate(self.agent, ActionChoice(name=name, arguments={}))
        return ArchetypePlan(step=obs.step, llm_plan=f"Archetype action: {name}",
                             ttl=ttl, action=choice, probabilities=probabilities,
                             archetype_key=key)

    def plan(self, prompt: str | None = None, obs: Observation | None = None,
             ttl: int = 1, tools: ToolSelection | object = _UNSET,
             tool_calls: str | None = "auto", selected_tools: ToolSelection | object = _UNSET):
        if tools is not _UNSET or selected_tools is not _UNSET:
            raise ValueError("Archetype reasoning doesn't support tool selection")
        if tool_calls not in ("auto", None):
            raise ValueError("Archetype plans do not generate tool calls")
        if prompt is not None or obs is not None or ttl != 1:
            raise ValueError("Staged archetype plan uses frozen inputs; no plan overrides")
        return self.manager.get_plan(self.agent)

    async def aplan(self, prompt: str | None = None, obs: Observation | None = None,
                    ttl: int = 1, tools: ToolSelection | object = _UNSET,
                    tool_calls: str | None = "auto", selected_tools: ToolSelection | object = _UNSET):
        if tools is not _UNSET or selected_tools is not _UNSET:
            raise ValueError("Archetype reasoning doesn't support tool selection")
        if tool_calls not in ("auto", None):
            raise ValueError("Archetype plans do not generate tool calls")
        if prompt is not None or obs is not None or ttl != 1:
            raise ValueError("Staged archetype plan uses frozen inputs; no plan overrides")
        return self.manager.get_plan(self.agent)

    def act(self, **kwargs):
        """Estimate/sample and execute a registered action without an extra LLM call."""
        return self.agent.apply_plan(self.plan(**kwargs))

    async def aact(self, **kwargs):
        return await self.agent.aapply_plan(await self.aplan(**kwargs))
