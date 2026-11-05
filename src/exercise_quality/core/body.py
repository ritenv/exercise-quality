from dataclasses import dataclass


@dataclass(frozen=True)
class BodyProportions:
    shin: float = 0.43
    thigh: float = 0.45
    torso: float = 0.55
    neck: float = 0.12
    foot: float = 0.25

    def __post_init__(self):
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )
