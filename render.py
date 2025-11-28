import json
import math
import os
import random

import cv2
import numpy as np
import pygame

# ===============================
# CONSTANTS
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
    base = random.randint(50, 150)

    def noise():
        return min(255, max(0, base + random.randint(-15, 15)))

    return (noise(), noise(), noise())


def random_light_gray():
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
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        points.append((int(x), int(y)))
    return points


def ellipse_border_point(cx, cy, w, h, target_x, target_y):
    """
    Returns the point on the border of the ellipse (cx,cy,w,h) towards target (target_x,target_y)
    """
    dx = target_x - cx
    dy = target_y - cy
    if dx == 0 and dy == 0:
        return cx, cy
    angle = math.atan2(dy, dx)
    rx = w / 2
    ry = h / 2
    border_x = cx + rx * math.cos(angle)
    border_y = cy + ry * math.sin(angle)
    return int(border_x), int(border_y)


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

        screen = pygame.Surface((width, height))
        screen.fill(BG_COLOR)

        # ---- draw edges ----
        for e in edges:
            v1_id, v2_id = e["relationship"]
            v1 = vertex_map[v1_id]
            v2 = vertex_map[v2_id]

            v1_cx = int(v1["x"] * width + v1["width"] * width / 2)
            v1_cy = int(v1["y"] * height + v1["height"] * height / 2)
            v2_cx = int(v2["x"] * width + v2["width"] * width / 2)
            v2_cy = int(v2["y"] * height + v2["height"] * height / 2)

            # start/end at ellipse border
            start = ellipse_border_point(
                v1_cx,
                v1_cy,
                int(v1["width"] * width),
                int(v1["height"] * height),
                v2_cx,
                v2_cy,
            )
            end = ellipse_border_point(
                v2_cx,
                v2_cy,
                int(v2["width"] * width),
                int(v2["height"] * height),
                v1_cx,
                v1_cy,
            )

            x = int(e["x"] * width)
            y = int(e["y"] * height)
            w = int(e["width"] * width)
            h = int(e["height"] * height)

            mid_x = x + w / 2
            mid_y = y + h / 2

            max_offset_x = w * MAX_CURVE_OFFSET_FACTOR
            max_offset_y = h * MAX_CURVE_OFFSET_FACTOR

            offset_x = random.uniform(-max(5, max_offset_x), max(5, max_offset_x))
            offset_y = random.uniform(-max(5, max_offset_y), max(5, max_offset_y))
            ctrl = (int(mid_x + offset_x), int(mid_y + offset_y))

            curve_points = get_bezier_points(start, ctrl, end, CURVE_RESOLUTION)
            if len(curve_points) > 1:
                pygame.draw.lines(
                    screen, EDGE_COLOR, False, curve_points, EDGE_THICKNESS
                )

                temp_edge_mask = np.zeros((height, width), dtype=np.uint8)
                for i in range(len(curve_points) - 1):
                    cv2.line(
                        temp_edge_mask,
                        curve_points[i],
                        curve_points[i + 1],
                        255,
                        EDGE_THICKNESS,
                    )

                polys = extract_contours(temp_edge_mask)

                for poly in polys:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    if DBG:
                        pygame.draw.rect(
                            screen,
                            (255, 0, 0),
                            pygame.Rect(
                                min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                            ),
                            2,
                        )

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

        # ---- draw nodes ----
        for v in vertices:
            x = int(v["x"] * width)
            y = int(v["y"] * height)
            w = int(v["width"] * width)
            h = int(v["height"] * height)
            rect = pygame.Rect(x, y, w, h)
            pygame.draw.ellipse(screen, NODE_COLOR, rect)
            pygame.draw.ellipse(screen, NODE_BORDER, rect, NODE_THICKNESS)

            if DBG:
                pygame.draw.rect(screen, (0, 0, 255), rect, 2)

            font_size = max(
                12, int(h * 0.6 / max(1, math.log(len(v["symbol"]) + 1, 3)))
            )
            font = pygame.font.SysFont("Arial", font_size)
            text = font.render(v["symbol"], True, (0, 0, 0))
            text_rect = text.get_rect(center=(x + w / 2, y + h / 2))
            screen.blit(text, text_rect)

            # Node mask for COCO
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

        screen = add_salt_pepper(screen)
        image_path = os.path.join(image_folder, img_filename)
        pygame.image.save(screen, image_path)

        coco["images"].append(
            {"id": img_id, "file_name": img_filename, "width": width, "height": height}
        )
        img_id += 1
        print(f"✓ {split}: {img_filename}")

    out_json = os.path.join(OUTPUT_ANNOTATIONS, f"{split}_coco.json")
    with open(out_json, "w") as f:
        json.dump(coco, f)
    print(f"🔥 Saved: {out_json}")

pygame.quit()
print("🎉 All COCO datasets generated successfully!")
