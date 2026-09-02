from dataclasses import asdict, dataclass


@dataclass
class ModelCkp:

    epoch: int
    phase: int
    models: dict[str, dict]
    optimizers: dict[str, dict]
    schedulers: dict[str, dict]
    config: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelCkp":
        return cls(**data)
