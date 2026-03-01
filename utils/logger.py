import json


def save_result(data, filename="output.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
