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

NODE_THICKNESS = 4
EDGE_THICKNESS = 5

SALT_PEPPER_AMOUNT = 0.005
SALT_RATIO = 0.6

CURVE_RESOLUTION = 30
MAX_CURVE_OFFSET_FACTOR = 0.3

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


def get_bezier_points(p0, p1, p2, num_points):
    """
    Calculates points along a quadratic Bézier curve.
    p0 (start), p1 (control), p2 (end) are tuples (x, y).
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        # Quadratic Bézier formula: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        points.append((int(x), int(y)))
    return points


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
        BG_COLOR = random_light_gray()
        NODE_COLOR = BG_COLOR
        NODE_BORDER = random_gray()
        EDGE_COLOR = random_gray()

        vertices = graph["vertices"]
        edges = graph["connections"]
        width = int(graph.get("img_width", 800))
        height = int(graph.get("img_height", 800))
        img_filename = graph["filename"]

        vertex_map = {v["id"]: v for v in vertices}

        # ---- create pygame surface ----
        screen = pygame.Surface((width, height))
        screen.fill(BG_COLOR)

        # NOTE: edge_mask removed. Segmentation is now performed per-edge.

        # ---- draw edges (CURVED) and generate isolated masks ----
        for e in edges:
            v1_id, v2_id = e["relationship"]
            v1_data = vertex_map[v1_id]
            v2_data = vertex_map[v2_id]

            # 1. Determine Start and End points (centers of the nodes)
            v1_center_x = int(v1_data["x"] * width + v1_data["width"] * width / 2)
            v1_center_y = int(v1_data["y"] * height + v1_data["height"] * height / 2)
            v2_center_x = int(v2_data["x"] * width + v2_data["width"] * width / 2)
            v2_center_y = int(v2_data["y"] * height + v2_data["height"] * height / 2)

            start = (v1_center_x, v1_center_y)
            end = (v2_center_x, v2_center_y)

            # 2. Calculate Control Point (C)
            # Edge bounding box in pixel coordinates
            x = int(e["x"] * width)
            y = int(e["y"] * height)
            w = int(e["width"] * width)
            h = int(e["height"] * height)

            mid_x = x + w / 2
            mid_y = y + h / 2

            # Random offset based on bounding box size
            max_offset_x = w * MAX_CURVE_OFFSET_FACTOR
            max_offset_y = h * MAX_CURVE_OFFSET_FACTOR

            # Ensure a minimum 5px offset for visible curvature, maxing out at the calculated factor
            offset_x = random.uniform(-max(5, max_offset_x), max(5, max_offset_x))
            offset_y = random.uniform(-max(5, max_offset_y), max(5, max_offset_y))

            # Control point is the center of the edge bbox + random offset
            ctrl = (int(mid_x + offset_x), int(mid_y + offset_y))

            # 3. Generate Bézier Curve Points
            curve_points = get_bezier_points(start, ctrl, end, CURVE_RESOLUTION)

            # 4. Draw curve on Pygame screen (Visual rendering)
            if len(curve_points) > 1:
                pygame.draw.lines(
                    screen, EDGE_COLOR, False, curve_points, EDGE_THICKNESS
                )

                # 5. Draw curve on TEMPORARY, ISOLATED OpenCV mask for annotation
                temp_edge_mask = np.zeros((height, width), dtype=np.uint8)
                for i in range(len(curve_points) - 1):
                    pt1 = curve_points[i]
                    pt2 = curve_points[i + 1]
                    cv2.line(temp_edge_mask, pt1, pt2, 255, EDGE_THICKNESS)

                # 6. Extract contours for this SINGLE edge instance and add to COCO
                polys = extract_contours(temp_edge_mask)
                for poly in polys:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    coco["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 2,
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

        # ---- draw nodes (to ensure nodes are drawn over edges) ----
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

        # ---- EDGE MASKS: Logic moved inside the edge loop to ensure isolation. The previous block is removed.

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
