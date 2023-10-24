# mlcopilot(config_space, task) -> config
# References:
# - same config space
# - other (similar) tasks
# - solutions with best metrics

from typing import Literal, Dict, Any, Optional, List

Config = Dict[str, Any]

class Condition:
    match: Optional[Config]

class Parameter:
    name: str
    dtype: Literal["int", "float", "str", "bool", "any"]
    categorical: bool
    choices: List[str]
    low: Optional[float]
    high: Optional[float]
    log_distributed: Optional[bool]
    condition: Optional[Condition]
    quantiles: Optional[float]

class Space:
    name: str
    description: str
    parameters: List[Parameter]

class Task:
    role: Literal["dataset", "type", "any"]
    description: str

class Solution:
    space: Space
    task: List[Task]
    config: Config
    cano_config: Config
    metric: Optional[float]
    feedback: Optional[str]

class Knowledge:
    space: Space
    task: List[Task]
    text: str
