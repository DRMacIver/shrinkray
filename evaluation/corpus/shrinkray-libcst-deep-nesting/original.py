"""Configuration builder for the widget pipeline.

The pipeline is assembled by composing a stack of wrapper stages; the
composed expression below is generated and can nest very deeply.
"""

from dataclasses import dataclass


def wrap(stage):
    """Wrap a pipeline stage, returning the next stage in the chain."""
    return stage


@dataclass
class Stage:
    name: str
    weight: int = 1

    def compose(self, other: "Stage") -> "Stage":
        return Stage(self.name + "+" + other.name, self.weight + other.weight)


DEFAULT_STAGES = [Stage("ingest"), Stage("normalise"), Stage("emit")]


def build_pipeline():
    """Build the composed pipeline value used by the runtime."""
    composed = wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(wrap(0))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
    return composed


if __name__ == "__main__":
    print(build_pipeline())
