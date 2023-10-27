import inspect
import json

import pandas as pd

from .data_manager import create_tables
from .schema import *


def read_json_as_dict(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {d["id"]: d for d in data}


def inject_hpob_data():
    schemas_raw = read_json_as_dict("coml/configagent/data/schemas.json")
    schemas = {}
    for k, v in schemas_raw.items():
        if not v["description"].startswith("Learner "):
            continue
        parameters = []
        for p in v["parameters"]:
            p_ = {}
            condition = p.pop("condition")
            if condition:
                p_["required"] = condition
            else:
                p_["required"] = True
            p_["log_distributed"] = p.pop("logDistributed")
            p_.update(p)
            parameters.append(p_)
        space = Space(
            id=k,
            name=v["description"].split()[1],
            description=v["description"],
            parameters=parameters,
        )
        schemas[k] = space

    datasets = read_json_as_dict("coml/configagent/data/datasets.json")
    datasets = {k: Context(
        id=v["id"],
        type="dataset",
        name=v["name"],
        description=v["description"],
    ) for k, v in datasets.items() if v["id"].startswith("openml-")}

    solutions = read_json_as_dict("coml/configagent/data/solutions.json")
    solutions = {k: Solution(
        id=v["id"],
        space=schemas[v["modules"][1]["module"]["schema"]],
        context=[datasets[v["modules"][0]["module"]]],
        config=v["modules"][1]["module"]["config"],
        metric=v["metrics"],
    ) for k, v in solutions.items() if v["id"].startswith("hpob-")}

    guidelines = read_json_as_dict("coml/configagent/data/knowledges.json")
    guidelines = {k: Guideline(
        id=v["id"],
        space=schemas[v["subjectSchema"]],
        context=None,
        guideline=v["knowledge"]
    ) for k, v in guidelines.items() if v["id"].startswith("hpob-")}

    Space.insert_many(schemas.values())
    Context.insert_many(datasets.values())
    Solution.insert_many(solutions.values())
    Guideline.insert_many(guidelines.values())

    list(Space.query_all())
    list(Context.query_all())
    list(Solution.query_all())
    list(Guideline.query_all())


def _import(name):
    module, comp = name.rsplit('.', 1)
    mod = __import__(module, fromlist=[comp])
    return getattr(mod, comp)


def prepare_sklearn_spaces():
    kaggle_data = pd.read_json("coml/configagent/data/kaggle.jsonl", lines=True)
    apis = list(kaggle_data["api"].unique())
    for api in apis:
        if "/" in api:
            continue

        params = []
        positional_counter = 0
        for param_name, param in inspect.signature(_import(api)).parameters.items():
            dtype = param.annotation
            if dtype in [int, float, str, bool]:
                dtype = dtype.__name__
            else:
                dtype = "any"
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                name = f"[{positional_counter}] ({param_name})"
                positional_counter += 1
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                name = param_name
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                name = f"[{positional_counter}:] ({param_name})"
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                name = f"**{param_name}"
            else:
                raise NotImplementedError()

            choices = None
            description = None
            ...
            params.append(Parameter(
                name=name,
                dtype=dtype,
                categorical=bool(choices),
                required=param.default == inspect.Parameter.empty,
                has_default=param.default != inspect.Parameter.empty,
                default_value=param.default if dtype != "any" or param.default is None else str(param.default),
                choices=choices,
                description=description,
                low=low,
                high=high,
                log_distributed=log_distributed,
                quantiles=quantiles,
            ))


#     instruction = """Given a docstring of a function, please summarize it in the following format:

# name: <name of the function>
# description: <description of the function> (in 1 - 3 sentences.)
# parameters:
#   <parameter name>:
#     dtype: <int|float|str|bool|any> (if the data type is not int, float, str or bool, use any)
#     categorical: <true|false> (if the parameter has a finite set of choices, use true, otherwise use false)
#     choices: <list of choices> (if the parameter is categorical, otherwise omit this line)
#     required: <true|false> (if the parameter is required, i.e., a positional argument, use true, otherwise use false)
#     description: <brief description of the parameter> (in 1 - 3 sentences.)
#     default_value: 
    """




# def inject_kaggle_data():



def main():
    create_tables()
    inject_hpob_data()

# main()
prepare_sklearn_spaces()

