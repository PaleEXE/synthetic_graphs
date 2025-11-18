import pygame
import json
import os
import math
import cv2
import numpy as np

INPUT_JSON = "synth_graphs_2.json"
OUTPUT_FOLDER = "synth_graphs_2"
OUTPUT_COCO = "synth_graphs_coco_2.json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BG_COLOR = (255, 250, 240)
NODE_COLOR = (255, 140, 66)
NODE_BORDER = (210, 90, 45)
EDGE_COLOR = (180, 80, 60)
COST_COLOR = (120, 60, 40)

DBG_EDGE = (255, 70, 70)
DBG_VERT = (70, 130, 180)
DBG = True  # Set True to draw debug bounding boxes

pygame.init()
clock = pygame.time.Clock()

data = json.load(open(INPUT_JSON))

coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "node"},
        {"id": 2, "name": "edge"}
    ]
}

ann_id = 1
img_id = 1


def polygon_area(xs, ys):
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 1:
            continue
        polys.append([(int(x), int(y)) for x, y in cnt])
    return polys


for graph in data:
    vertices = graph["vertices"]
    edges = graph["connections"]
    width = int(graph.get("img_width", 800))
    height = int(graph.get("img_height", 800))

    # 1. Setup Pygame surface
    screen = pygame.Surface((width, height))
    screen.fill(BG_COLOR)

    # 2. Draw edges
    for e in edges:
        x = e["x"] * width
        y = e["y"] * height
        w = e["width"] * width
        h = e["height"] * height
        v1, v2 = e["relationship"]
        from_top = (
            vertices[v1]["y"] < vertices[v2]["y"]
            if vertices[v1]["x"] <= vertices[v2]["x"]
            else vertices[v1]["y"] > vertices[v2]["y"]
        )
        if from_top:
            start_pos = (int(x), int(y))
            end_pos = (int(x + w), int(y + h))
        else:
            start_pos = (int(x), int(y + h))
            end_pos = (int(x + w), int(y))

        # Draw edge
        pygame.draw.line(screen, EDGE_COLOR, start_pos, end_pos, 4)

        # --- DEBUG: draw edge bounding box ---
        if DBG:
            min_x = min(start_pos[0], end_pos[0])
            max_x = max(start_pos[0], end_pos[0])
            min_y = min(start_pos[1], end_pos[1])
            max_y = max(start_pos[1], end_pos[1])
            edge_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            pygame.draw.rect(screen, DBG_EDGE, edge_rect, 2)

    # 3. Draw vertices AND symbols
    for v in vertices:
        x = v["x"] * width
        y = v["y"] * height
        w = v["width"] * width
        h = v["height"] * height
        node_rect = pygame.Rect(x, y, w, h)
        pygame.draw.ellipse(screen, NODE_COLOR, node_rect)
        pygame.draw.ellipse(screen, NODE_BORDER, node_rect, 2)

        # --- DEBUG: draw vertex bounding box ---
        if DBG:
            pygame.draw.rect(screen, DBG_VERT, node_rect, 2)

        # Draw symbol
        font_size = max(12, int(h * 0.6 / max(1, math.log(len(v["symbol"]) + 1, 3))))
        font = pygame.font.SysFont("Arial", font_size)
        text = font.render(v["symbol"], True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + w / 2, y + h / 2))
        screen.blit(text, text_rect)

    # 4. Save image
    img_filename = graph["filename"]
    pygame.image.save(screen, os.path.join(OUTPUT_FOLDER, img_filename))

    # Convert Pygame surface to OpenCV image for contours
    img_array = pygame.surfarray.array3d(screen)
    img_cv = cv2.cvtColor(np.transpose(img_array, (1, 0, 2)), cv2.COLOR_RGB2BGR)

    # -------------------
    # Nodes (category 1)
    # -------------------
    for v in vertices:
        x = int(v["x"] * width)
        y = int(v["y"] * height)
        w = int(v["width"] * width)
        h = int(v["height"] * height)
        node_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(node_mask, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        polys = extract_contours(node_mask)
        for poly in polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [[int(c) for pt in poly for c in pt]],
                "bbox": [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
                "area": float(polygon_area(xs, ys)),
                "iscrowd": 0
            })
            ann_id += 1

    # -------------------
    # Edges (category 2)
    # -------------------
    edge_mask = np.zeros((height, width), dtype=np.uint8)
    for e in edges:
        x = e["x"] * width
        y = e["y"] * height
        w = e["width"] * width
        h = e["height"] * height
        v1, v2 = e["relationship"]
        from_top = (
            vertices[v1]["y"] < vertices[v2]["y"]
            if vertices[v1]["x"] <= vertices[v2]["x"]
            else vertices[v1]["y"] > vertices[v2]["y"]
        )
        if from_top:
            start_pos = (int(x), int(y))
            end_pos = (int(x + w), int(y + h))
        else:
            start_pos = (int(x), int(y + h))
            end_pos = (int(x + w), int(y))

        cv2.line(edge_mask, start_pos, end_pos, 255, 4)

    polys = extract_contours(edge_mask)
    for poly in polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 2,
            "segmentation": [[int(c) for pt in poly for c in pt]],
            "bbox": [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
            "area": float(polygon_area(xs, ys)),
            "iscrowd": 0
        })
        ann_id += 1

    # -------------------
    # Image info
    # -------------------
    coco["images"].append({
        "id": img_id,
        "file_name": img_filename,
        "width": width,
        "height": height
    })

    img_id += 1
    print(f"✅ Processed {img_filename}")

# Save COCO JSON
with open(OUTPUT_COCO, "w") as f:
    json.dump(coco, f)

pygame.quit()
print(f"🎉 COCO annotations saved to {OUTPUT_COCO}")
