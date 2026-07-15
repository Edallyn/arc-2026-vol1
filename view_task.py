import json
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
CMAP = ListedColormap(ARC_COLORS)

with open("data/arc-prize-2024/arc-agi_evaluation_challenges.json") as f:
    challenges = json.load(f)

task_id = sys.argv[1]
task = challenges[task_id]

train = task["train"]
test = task["test"]
n_rows = len(train) + len(test)

fig, axes = plt.subplots(n_rows, 2, figsize=(5, 2.5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, 2)

def draw(ax, grid, title):
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

row = 0
for i, ex in enumerate(train):
    draw(axes[row][0], ex["input"], f"train[{i}] input")
    draw(axes[row][1], ex["output"], f"train[{i}] output")
    row += 1

for i, ex in enumerate(test):
    draw(axes[row][0], ex["input"], f"test[{i}] input")
    axes[row][1].axis("off")
    axes[row][1].set_title(f"test[{i}] output = ?", fontsize=9)
    row += 1

fig.suptitle(f"Task: {task_id}", fontsize=12)
plt.tight_layout()

out_path = f"task_{task_id}.png"
plt.savefig(out_path, dpi=130)
print(f"Kaydedildi: {out_path}")