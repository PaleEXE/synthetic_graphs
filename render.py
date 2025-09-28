import json
import matplotlib.pyplot as plt

# Suppose you dumped your vertices/edges from Rust as JSON
with open("graph.json") as f:
    data = json.load(f)

vertices = data["vertices"]
edges = data["edges"]

fig, ax = plt.subplots()

# draw edges (beneath nodes)
for e in edges:
    v1 = next(v for v in vertices if v["id"] == e["relationship"][0])
    v2 = next(v for v in vertices if v["id"] == e["relationship"][1])
    ax.plot(
        [v1["x"], v2["x"]], [v1["y"], v2["y"]],
        color="gray", linewidth=1.5, alpha=0.7, zorder=1
    )

# draw nodes (on top of edges)
for v in vertices:
    circle = plt.Circle(
        (v["x"], v["y"]),
        v["width"] / 2,
        facecolor="pink",
        edgecolor="pink",
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(circle)
    ax.text(
        v["x"], v["y"], v["symbol"],
        ha="center", va="center",
        fontsize=v["width"] / 4, color="black", weight="bold",
        zorder=3
    )

ax.set_aspect("equal")
ax.axis("off")  # hide axes for a clean look
plt.show()
