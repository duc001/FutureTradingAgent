
import json

a = {
    "id": 1,
    "name": "李四",
    "address": {
        "city": "北京",
        "street": "长安街"
    },
    "hobbies": ["读书", "游泳", "编程"],
    "friends": [
        {"name": "王五", "age": 26},
        {"name": "赵六", "age": 24}
    ]
}

with open("a.json", "w") as f:
    json.dump(a, f, ensure_ascii=False, indent=4)

