"""Agent adapters: one per harness, all satisfying AgentAdapter."""

from earshot.agents.base import AgentAdapter, AgentError
from earshot.config import AgentConfig, Config


def create_adapter(name: str, agent_config: AgentConfig) -> AgentAdapter:
    """Instantiate the adapter for one configured agent."""
    if agent_config.harness == "opencode":
        from earshot.agents.opencode import OpencodeAdapter

        return OpencodeAdapter(name, agent_config)
    # config validation guarantees harness is one of the known three
    raise NotImplementedError(
        f"the {agent_config.harness!r} adapter is issue #9; only 'opencode' works today"
    )


def first_agent(config: Config) -> tuple[str, AgentConfig]:
    """The single-agent Phase 1 rule: the first configured agent is used."""
    name = next(iter(config.agents))
    return name, config.agents[name]


__all__ = ["AgentAdapter", "AgentError", "create_adapter", "first_agent"]
