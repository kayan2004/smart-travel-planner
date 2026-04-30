from app.agent.tools.base import BaseTool
from app.agent.tools.classifier_tool import TravelStyleClassifierTool
from app.agent.tools.live_conditions_tool import LiveConditionsTool
from app.agent.tools.rag_retrieval_tool import DestinationContextRetrieverTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> BaseTool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"Tool '{tool_name}' is not registered.") from exc

    def list_names(self) -> list[str]:
        return sorted(self._tools)


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(TravelStyleClassifierTool())
    registry.register(DestinationContextRetrieverTool())
    registry.register(LiveConditionsTool())
    return registry

