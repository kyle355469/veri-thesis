from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Protocol


class PipelineStage(Protocol):
    name: str

    def __call__(self, context: MutableMapping[str, Any]) -> Any:
        ...


StageCallable = Callable[[MutableMapping[str, Any]], Any]
StageCallback = Callable[[Dict[str, Any]], None]


@dataclass(frozen=True)
class FunctionStage:
    name: str
    fn: StageCallable

    def __call__(self, context: MutableMapping[str, Any]) -> Any:
        return self.fn(context)


class AgenticPipeline:
    """Composable stage runner for trying different agentic IP reuse flows."""

    def __init__(
        self,
        stages: Optional[Iterable[PipelineStage | tuple[str, StageCallable] | StageCallable]] = None,
        *,
        stage_callback: Optional[StageCallback] = None,
    ) -> None:
        self.stages: List[PipelineStage] = []
        self.stage_callback = stage_callback
        for stage in stages or []:
            self.add_stage(stage)

    def add_stage(
        self,
        stage: PipelineStage | tuple[str, StageCallable] | StageCallable,
        name: Optional[str] = None,
    ) -> "AgenticPipeline":
        if isinstance(stage, tuple):
            stage_name, fn = stage
            self.stages.append(FunctionStage(stage_name, fn))
            return self
        if name is not None:
            self.stages.append(FunctionStage(name, stage))
            return self
        if hasattr(stage, "name"):
            self.stages.append(stage)
            return self
        stage_name = getattr(stage, "__name__", f"stage_{len(self.stages) + 1}")
        self.stages.append(FunctionStage(stage_name, stage))
        return self

    def extend(
        self,
        stages: Iterable[PipelineStage | tuple[str, StageCallable] | StageCallable],
    ) -> "AgenticPipeline":
        for stage in stages:
            self.add_stage(stage)
        return self

    def run(self, initial_context: Optional[MutableMapping[str, Any]] = None) -> MutableMapping[str, Any]:
        context = initial_context or {}
        for stage in self.stages:
            self._emit(stage.name, "running")
            result = stage(context)
            if isinstance(result, MutableMapping):
                context.update(result)
            elif result is not None:
                context[stage.name] = result
            self._emit(stage.name, "complete")
        return context

    def _emit(self, stage: str, status: str, **payload: Any) -> None:
        if self.stage_callback is None:
            return
        self.stage_callback({"stage": stage, "status": status, **payload})
