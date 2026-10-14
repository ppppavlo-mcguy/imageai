# instruction: python that fetches json from an api
import json
import urllib.request


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    data = get_json("https://api.github.com/repos/python/cpython")
    print(data["full_name"], data["stargazers_count"])
