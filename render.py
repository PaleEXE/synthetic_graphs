import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Load JSON
with open("graph.json") as f:
    data = json.load(f)

vertices = data["vertices"]
edges = data["edges"]

# Get exact dimensions from JSON
img_width = data["img_width"]
img_height = data["img_height"]

# Create figure with exact resolution
fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

# Draw edges first (beneath nodes)
for e in edges:
    v1 = next(v for v in vertices if v["id"] == e["relationship"][0])
    v2 = next(v for v in vertices if v["id"] == e["relationship"][1])

    # Use actual vertex centers for edge endpoints
    x1 = (v1["x"] + v1["width"] / 2) * img_width
    y1 = (v1["y"] + v1["height"] / 2) * img_height
    x2 = (v2["x"] + v2["width"] / 2) * img_width
    y2 = (v2["y"] + v2["height"] / 2) * img_height

    ax.plot([x1, x2], [y1, y2], color="gray", linewidth=2, alpha=0.7, zorder=1)

# Draw vertices
for v in vertices:
    center_x = (v["x"] + v["width"] / 2) * img_width
    center_y = (v["y"] + v["height"] / 2) * img_height
    radius = v["width"] * img_width / 2

    circle = Circle(
        (center_x, center_y),
        radius,
        facecolor="lightblue",
        edgecolor="darkblue",
        linewidth=2,
        zorder=2
    )
    ax.add_patch(circle)

    ax.text(
        center_x, center_y, v["symbol"],
        ha="center", va="center",
        fontsize=12, color="black", weight="bold",
        zorder=3
    )

# Set exact limits matching image dimensions
ax.set_xlim(-(vertices[0]["width"] / 2) * img_width, img_width - (vertices[0]["width"] / 2) * img_width)
ax.set_ylim(-(vertices[0]["height"] / 2) * img_height, img_height - (vertices[0]["height"] / 2) * img_height)
ax.set_aspect("equal")

# Flip y-axis to match typical coordinate systems
ax.invert_yaxis()
ax.axis("off")

# Remove padding and save with exact dimensions
plt.tight_layout(pad=0)
plt.savefig("graph.png", dpi=100, bbox_inches='tight', pad_inches=0)
plt.show()
