from dataclasses import dataclass, field
import enum
from typing import List

from pldm.objectives.vicreg import VICRegObjective, VICRegObjectiveConfig  # noqa
from pldm.objectives.idm import IDMObjective, IDMObjectiveConfig  # noqa
from pldm.objectives.prediction import PredictionObjective, PredictionObjectiveConfig


class ObjectiveType(enum.Enum):
    VICReg = enum.auto()
    VICRegObs = enum.auto()
    VICRegPropio = enum.auto()
    IDM = enum.auto()
    Prediction = enum.auto()
    PredictionObs = enum.auto()
    PredictionPropio = enum.auto()


@dataclass
class ObjectivesConfig:
    objectives: List[ObjectiveType] = field(default_factory=lambda: [])
    vicreg: VICRegObjectiveConfig = VICRegObjectiveConfig()
    vicreg_obs: VICRegObjectiveConfig = VICRegObjectiveConfig()
    vicreg_propio: VICRegObjectiveConfig = VICRegObjectiveConfig()
    idm: IDMObjectiveConfig = IDMObjectiveConfig()
    prediction: PredictionObjectiveConfig = PredictionObjectiveConfig()
    prediction_obs: PredictionObjectiveConfig = PredictionObjectiveConfig()
    prediction_propio: PredictionObjectiveConfig = PredictionObjectiveConfig()

    def build_objectives_list(
        self,
        repr_dim: int,
        name_prefix: str = "",
    ):
        objectives = []
        for objective_type in self.objectives:
            if objective_type == ObjectiveType.VICReg:
                objectives.append(
                    VICRegObjective(
                        self.vicreg, name_prefix=name_prefix, repr_dim=repr_dim
                    )
                )
            elif objective_type == ObjectiveType.VICRegObs:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.VICRegPropio:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_propio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="propio",
                    )
                )
            elif objective_type == ObjectiveType.IDM:
                objectives.append(
                    IDMObjective(self.idm, name_prefix=name_prefix, repr_dim=repr_dim)
                )
            elif objective_type == ObjectiveType.PredictionObs:
                objectives.append(
                    PredictionObjective(
                        self.prediction_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.PredictionPropio:
                objectives.append(
                    PredictionObjective(
                        self.prediction_propio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="propio",
                    )
                )
            else:
                raise NotImplementedError()
        return objectives
