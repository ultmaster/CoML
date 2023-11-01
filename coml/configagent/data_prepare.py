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


def _strip_optional(s):
    if str(s).startswith("typing.Optional["):
        return s.__args__[0], True
    return s, False

def _print_value(value):
    import numpy, pandas
    if isinstance(value, numpy.ndarray):
        return 'array(shape={})'.format(value.shape)
    elif isinstance(value, pandas.DataFrame):
        return 'dataframe(shape={})'.format(value.shape)
    elif isinstance(value, pandas.Series):
        return 'series(shape={})'.format(value.shape)
    elif isinstance(value, list):
        if len(value) > 30:
            return '[{}, ...]'.format(', '.join(_print_value(v) for v in value[:30]))
        return '[{}]'.format(', '.join(_print_value(v) for v in value))
    elif isinstance(value, dict):
        if len(value) > 30:
            return '{{{}, ...}}'.format(', '.join(f'{k}: {_print_value(v)}' for k, v in list(value.items())[:30]))
        return '{{{}}}'.format(', '.join(f'{k}: {_print_value(v)}' for k, v in value.items()))
    elif isinstance(value, (str, bool, int, float)):
        return value
    elif value is None:
        return None
    else:
        val = str(value)
        if len(val) > 100:
            val = val[:100] + '...'
        return val


def prepare_sklearn_spaces():
    kaggle_data = pd.read_json("coml/configagent/data/kaggle.jsonl", lines=True)
    apis = list(kaggle_data["api"].unique())

    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage

    import dotenv
    dotenv.load_dotenv()

    llm = AzureChatOpenAI(max_retries=10, temperature=0.)
    system_message = SystemMessage(content="""
You will be given a docstring of a function. Please summarize it in the following YAML format:
                                   
<parameter 0 name>:
  choices: <list of choices>   # if the parameter has a limited number of choices, otherwise omit this line
  low: <number>                # if the parameter is a number and its lower bound is mentioned in the docstring, otherwise omit this line
  high: <number>               # if the parameter is a number and its upper bound is mentioned in the docstring, otherwise omit this line
  description: <string>        # write a description of this parameter, in 1-3 sentences
<parameter 1 name>:
  ...                       
"""
    )


    results = {}

    for api in apis:
        if "/" in api:
            continue

        params = []
        parameters = kaggle_data[kaggle_data["api"] == api]["parameters"].tolist()
        for param_name, param in inspect.signature(_import(api)).parameters.items():
            dtype, nullable = _strip_optional(param.annotation)
            if dtype in [int, float, str, bool]:
                dtype = dtype.__name__
            else:
                dtype = "any"
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_ONLY):
                name = param_name
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                name = f"*{param_name}"
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                name = f"**{param_name}"
            else:
                raise NotImplementedError()

            default_value = None
            if param.default is not inspect.Parameter.empty:
                default_value = _print_value(param.default)

            empirical_values = [p[name] for p in parameters if name in p]
            if param.default is not inspect.Parameter.empty:
                empirical_values.append(default_value)
            if any(e in ("None", None) for e in empirical_values):
                nullable = True
                empirical_values = [e for e in empirical_values if e not in ("None", None)]

            choices = None
            description = None
            low = None
            high = None

            log_distributed = None
            quantiles = None
    
            if dtype in ["int", "float"]:
                log_distributed = False
                
                if len(empirical_values) >= 3:
                    # quantiles = np.quantile(empirical_values, [i / 20 for i in range(21)]).tolist()
                    quantiles = []
                if len(empirical_values) >= 5:
                    low, medium, high = np.quantile(empirical_values, [0.2, 0.5, 0.8])
                    if min(empirical_values) > 0 and medium != low and medium != high and \
                        abs((medium - low) / (high - medium) - 1) > 0.7 and \
                        abs((np.log(medium) - np.log(low)) / (np.log(high) - np.log(medium)) - 1) < 0.5:
                        log_distributed = True

            params.append(Parameter(
                name=name,
                dtype=dtype,
                categorical=bool(choices),
                required=True,  # sklearn can't have unset arguments
                nullable=nullable,
                has_default=param.default is not inspect.Parameter.empty,
                default_value=default_value,
                choices=choices,
                description=description,
                low=low,
                high=high,
                log_distributed=log_distributed,
                quantiles=quantiles,
            ))

        results[api] = [asdict(p) for p in params]

    with open("coml/configagent/data/sklearn_spaces.json", "w") as f:
        json.dump(results, f, indent=2)


# def inject_kaggle_data():



def main():
    create_tables()
    inject_hpob_data()

# main()
prepare_sklearn_spaces()

