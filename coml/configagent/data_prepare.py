import json
from .data_manager import create_tables
from .schema import *


def read_json_as_dict(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return {d["id"]: d for d in data}


def inject_hpob_data():
    schemas = read_json_as_dict("coml/configagent/data/schemas.json")
    schemas = {k: Space(
        id=k,
        name=v["description"].split()[1],
        description=v["description"],
        parameters=[
            Parameter(**{x.replace("logDistributed", "log_distributed"): y for x, y in p.items()}) for p in v["parameters"]
        ]
    ) for k, v in schemas.items() if v["description"].startswith("Learner ")}

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


# def inject_kaggle_data():



def main():
    create_tables()
    inject_hpob_data()

main()

