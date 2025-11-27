import json
import math
import os
import random

import cv2
import numpy as np
import pygame

# ===============================
# CONSTANTS (ALL IMPORTANT VALUES)
# ===============================
SPLITS = ["train", "val", "test"]
INPUT_JSONS = {"train": "train.json", "val": "val.json", "test": "test.json"}

OUTPUT_IMAGES = "images"
OUTPUT_ANNOTATIONS = "annotations"

NODE_THICKNESS = 2
EDGE_THICKNESS = 2

SALT_PEPPER_AMOUNT = 0.015  # % noise
SALT_RATIO = 0.6  # 60% white / 40% black

DBG = False
pygame.init()


# ===============================
# COLOR FUNCTIONS
# ===============================
def random_gray():
    """Random gray color for nodes/edges"""
    base = random.randint(50, 150)

    def noise():
        return min(255, max(0, base + random.randint(-15, 15)))

    return (noise(), noise(), noise())


def random_light_gray():
    """Light gray background"""
    base = random.randint(200, 255)

    def noise():
        return min(255, max(0, base + random.randint(-10, 10)))

    return (noise(), noise(), noise())


# ===============================
# UTILS
# ===============================
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


def add_salt_pepper(surf):
    """Add salt-and-pepper noise to pygame surface safely."""
    arr = pygame.surfarray.array3d(surf)
    arr = np.transpose(arr, (1, 0, 2))
    h, w, _ = arr.shape
    num_noise = int(SALT_PEPPER_AMOUNT * w * h)

    for _ in range(int(num_noise * SALT_RATIO)):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        arr[y, x] = [255, 255, 255]

    for _ in range(int(num_noise * (1 - SALT_RATIO))):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        arr[y, x] = [0, 0, 0]

    arr = np.transpose(arr, (1, 0, 2))
    return pygame.surfarray.make_surface(arr)


# ===============================
# CREATE OUTPUT FOLDERS
# ===============================
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_ANNOTATIONS, exist_ok=True)

# ===============================
# PROCESS SPLITS
# ===============================
for split in SPLITS:
    input_json = INPUT_JSONS[split]
    image_folder = os.path.join(OUTPUT_IMAGES, split)
    os.makedirs(image_folder, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "node"}, {"id": 2, "name": "edge"}],
    }

    ann_id = 1
    img_id = 1

    try:
        data = json.load(open(input_json))
    except FileNotFoundError:
        print(f"⚠ {input_json} missing -> skipping")
        continue

    for graph in data:
        # random colors per image
        BG_COLOR = random_light_gray()
        NODE_COLOR = BG_COLOR
        NODE_BORDER = random_gray()
        EDGE_COLOR = random_gray()

        vertices = graph["vertices"]
        edges = graph["connections"]
        width = int(graph.get("img_width", 800))
        height = int(graph.get("img_height", 800))
        img_filename = graph["filename"]

        # ---- create pygame surface ----
        screen = pygame.Surface((width, height))
        screen.fill(BG_COLOR)
        edge_mask = np.zeros((height, width), dtype=np.uint8)

        # ---- draw edges ----
        for e in edges:
            x = int(e["x"] * width)
            y = int(e["y"] * height)
            w = int(e["width"] * width)
            h = int(e["height"] * height)
            v1, v2 = e["relationship"]

            from_top = (
                vertices[v1]["y"] < vertices[v2]["y"]
                if vertices[v1]["x"] <= vertices[v2]["x"]
                else vertices[v1]["y"] > vertices[v2]["y"]
            )

            start = (x, y) if from_top else (x, y + h)
            end = (x + w, y + h) if from_top else (x + w, y)

            pygame.draw.line(screen, EDGE_COLOR, start, end, EDGE_THICKNESS)
            cv2.line(edge_mask, start, end, 255, EDGE_THICKNESS)

        # ---- draw nodes ----
        for v in vertices:
            x = int(v["x"] * width)
            y = int(v["y"] * height)
            w = int(v["width"] * width)
            h = int(v["height"] * height)
            rect = pygame.Rect(x, y, w, h)
            pygame.draw.ellipse(screen, NODE_COLOR, rect)
            pygame.draw.ellipse(screen, NODE_BORDER, rect, NODE_THICKNESS)

            # symbol text
            font_size = max(
                12, int(h * 0.6 / max(1, math.log(len(v["symbol"]) + 1, 3)))
            )
            font = pygame.font.SysFont("Arial", font_size)
            text = font.render(v["symbol"], True, (0, 0, 0))
            text_rect = text.get_rect(center=(x + w / 2, y + h / 2))
            screen.blit(text, text_rect)

        # ---- noise ----
        screen = add_salt_pepper(screen)
        # ---- save image ----
        image_path = os.path.join(image_folder, img_filename)
        pygame.image.save(screen, image_path)

        # ---- NODE MASKS ----
        for v in vertices:
            x = int(v["x"] * width)
            y = int(v["y"] * height)
            w = int(v["width"] * width)
            h = int(v["height"] * height)
            node_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                node_mask,
                (x + w // 2, y + h // 2),
                (w // 2, h // 2),
                0,
                0,
                360,
                255,
                -1,
            )
            polys = extract_contours(node_mask)
            for poly in polys:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": [[c for pt in poly for c in pt]],
                        "bbox": [
                            min(xs),
                            min(ys),
                            max(xs) - min(xs),
                            max(ys) - min(ys),
                        ],
                        "area": polygon_area(xs, ys),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        # ---- EDGE MASKS ----
        polys = extract_contours(edge_mask)
        for poly in polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 2,
                    "segmentation": [[c for pt in poly for c in pt]],
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "area": polygon_area(xs, ys),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        # ---- image info ----
        coco["images"].append(
            {"id": img_id, "file_name": img_filename, "width": width, "height": height}
        )

        img_id += 1
        print(f"✓ {split}: {img_filename}")

    # ---- save COCO JSON ----
    out_json = os.path.join(OUTPUT_ANNOTATIONS, f"{split}_coco.json")
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"🔥 Saved: {out_json}")


pygame.quit()
print("🎉 All COCO datasets generated successfully!")
