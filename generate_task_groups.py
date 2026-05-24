import json
import os

DATA_PATH = os.path.join(
    "data",
    "arc-prize-2024",
    "arc-agi_evaluation_challenges.json"
)

GROUP_SIZE = 20

with open(DATA_PATH, "r", encoding="utf-8") as f:
    challenges = json.load(f)

task_ids = list(challenges.keys())

groups = [
    task_ids[i:i + GROUP_SIZE]
    for i in range(0, len(task_ids), GROUP_SIZE)
]

print(f"Total tasks: {len(task_ids)}")
print(f"Total groups: {len(groups)}\n")

for i, g in enumerate(groups):
    print(f"# GROUP {i+1}")
    print("SELECTED_PROBLEMS = [")
    for t in g:
        print(f'    "{t}",')
    print("]\n")