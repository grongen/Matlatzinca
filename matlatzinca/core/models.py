import logging
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel as PydanticBaseModel
from pydantic.types import confloat
from pydantic import Field

logger = logging.getLogger(__name__)

subscript_dct = {
    "0": "\u2080",
    "1": "\u2081",
    "2": "\u2082",
    "3": "\u2083",
    "4": "\u2084",
    "5": "\u2085",
    "6": "\u2086",
    "7": "\u2087",
    "8": "\u2088",
    "9": "\u2089",
}


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        # use_enum_values = True
        extra = "forbid"  # will throw errors so we can fix our models
        # allow_population_by_field_name = True
        # alias_generator = underscore_to_space


class Edge(BaseModel):
    parent: str
    child: str
    string: str = None
    cond_rank_corr: confloat(ge=-1.0, le=1.0) = 0.0
    rank_corr: confloat(ge=-1.0, le=1.0) = np.nan
    rank_corr_bounds: Tuple[confloat(ge=-1.0, le=1.0), confloat(ge=-1.0, le=1.0)] = (np.nan, np.nan)

    @property
    def rank_corr_range(self):
        return f"({self.rank_corr_bounds[0]:.3g}, {self.rank_corr_bounds[1]:.3g})"

    @property
    def direction_string(self):
        return f"{self.parent} \u2192 {self.child}"

    @property
    def cond_rstring(self):
        subscript_string = self.string.replace("_", "")
        subscript_string = "".join(
            [subscript_dct[c] if c in subscript_dct else c for i, c in enumerate(subscript_string)]
        )
        return subscript_string

    @property
    def uncond_rstring(self):
        return self.cond_rstring.split("|")[0]


def get_random_01():
    return np.random.rand(1)[0]


class Node(BaseModel):
    name: str
    edges: List[Edge]
    x: float = Field(default_factory=get_random_01)
    y: float = Field(default_factory=get_random_01)

    @property
    def parent_names(self):
        return [edge.parent for edge in self.edges]

    def parent_index(self, parent_name: str) -> int:
        return self.parent_names.index(parent_name)

    def get_edge_by_parent(self, parent_name: str) -> Edge:
        return self.edges[self.parent_index(parent_name)]
