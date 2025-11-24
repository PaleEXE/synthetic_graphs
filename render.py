import pygame
import json
import os
import math
import cv2
import numpy as np
import random

GENERATE_YOLO_LABELS = True
INPUT_JSON = "dbg.json"
OUTPUT_FOLDER = "dbg"

BG_COLOR = (255, 250, 240)
DBG = True

def get_random_color():
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    return (r, g, b)

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

class GraphDrawer:
    """
    Modified to accept and use single, consistent colors for all nodes and edges.
    """
    def __init__(self, width, height, vertices, edges, node_color, edge_color):
        self.width = width
        self.height = height
        self.vertices = vertices
        self.edges = edges
        self.screen = pygame.Surface((self.width, self.height))
        self.screen.fill(BG_COLOR)

        # Stored fixed colors for the entire image
        self.NODE_COLOR = node_color
        self.NODE_BORDER = (node_color[0] // 2, node_color[1] // 2, node_color[2] // 2)
        self.EDGE_COLOR = edge_color

        self.node_bboxes = []
        self.edge_bboxes = []
        self.edge_mask = np.zeros((self.height, self.width), dtype=np.uint8)

    def _draw_edges(self):
        for e in self.edges:
            x = e["x"] * self.width
            y = e["y"] * self.height
            w = e["width"] * self.width
            h = e["height"] * self.height
            v1, v2 = e["relationship"]

            # Use the fixed self.EDGE_COLOR
            edge_color = self.EDGE_COLOR

            from_top = (
                self.vertices[v1]["y"] < self.vertices[v2]["y"]
                if self.vertices[v1]["x"] <= self.vertices[v2]["x"]
                else self.vertices[v1]["y"] > self.vertices[v2]["y"]
            )
            if from_top:
                start_pos = (int(x), int(y))
                end_pos = (int(x + w), int(y + h))
            else:
                start_pos = (int(x), int(y + h))
                end_pos = (int(x + w), int(y))

            pygame.draw.line(self.screen, edge_color, start_pos, end_pos, 4)
            cv2.line(self.edge_mask, start_pos, end_pos, 255, 4)

            min_x = max(0, min(start_pos[0], end_pos[0]) - 2)
            max_x = min(self.width, max(start_pos[0], end_pos[0]) + 2)
            min_y = max(0, min(start_pos[1], end_pos[1]) - 2)
            max_y = min(self.height, max(start_pos[1], end_pos[1]) + 2)

            if max_x > min_x and max_y > min_y:
                self.edge_bboxes.append((min_x, min_y, max_x, max_y))

            if DBG:
                edge_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
                pygame.draw.rect(self.screen, (255, 70, 70), edge_rect, 2)


    def _draw_vertices(self):
        for v in self.vertices:
            x = v["x"] * self.width
            y = v["y"] * self.height
            w = v["width"] * self.width
            h = v["height"] * self.height
            node_rect = pygame.Rect(x, y, w, h)

            # Use the fixed self.NODE_COLOR and self.NODE_BORDER
            node_color = self.NODE_COLOR
            node_border = self.NODE_BORDER

            pygame.draw.ellipse(self.screen, node_color, node_rect)
            pygame.draw.ellipse(self.screen, node_border, node_rect, 2)

            self.node_bboxes.append((x, y, x + w, y + h))

            font_size = max(12, int(h * 0.6 / max(1, math.log(len(v["symbol"]) + 1, 3))))
            font = pygame.font.SysFont("Arial", font_size)
            text = font.render(v["symbol"], True, (0, 0, 0))
            text_rect = text.get_rect(center=(x + w / 2, y + h / 2))
            self.screen.blit(text, text_rect)

            if DBG:
                pygame.draw.rect(self.screen, (70, 130, 180), node_rect, 2)

    def draw(self, output_path):
        self._draw_edges()
        self._draw_vertices()
        pygame.image.save(self.screen, output_path)

        return self.node_bboxes, self.edge_bboxes, self.edge_mask


class AnnotationGenerator:
    def __init__(self, generate_yolo, output_folder):
        self.generate_yolo = generate_yolo
        self.output_folder = output_folder
        self.base_folder = os.path.abspath(output_folder)

        if self.generate_yolo:
            self.yolo_classes = {"node": 0, "edge": 1}
            self.yolo_label_folder = os.path.join(output_folder, "labels")
            os.makedirs(self.yolo_label_folder, exist_ok=True)
            with open(os.path.join(output_folder, "classes.txt"), "w") as f:
                f.write("\n".join(self.yolo_classes.keys()))
        else:
            self.coco_output_file = os.path.join(output_folder, "val_coco.json")
            self.coco = {
                "images": [],
                "annotations": [],
                "categories": [{"id": 1, "name": "node"}, {"id": 2, "name": "edge"}]
            }
            self.ann_id = 1
            self.img_id = 1

    def _generate_yolo_for_image(self, filename, width, height, node_bboxes, edge_bboxes):
        yolo_annotations = []

        def to_yolo_format(class_id, min_x, min_y, max_x, max_y):
            x_center = ((min_x + max_x) / 2) / width
            y_center = ((min_y + max_y) / 2) / height
            w = (max_x - min_x) / width
            h = (max_y - min_y) / height
            return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        for min_x, min_y, max_x, max_y in node_bboxes:
            yolo_annotations.append(
                to_yolo_format(self.yolo_classes["node"], min_x, min_y, max_x, max_y)
            )

        for min_x, min_y, max_x, max_y in edge_bboxes:
            yolo_annotations.append(
                to_yolo_format(self.yolo_classes["edge"], min_x, min_y, max_x, max_y)
            )

        base_filename = os.path.splitext(filename)[0]
        label_path = os.path.join(self.yolo_label_folder, base_filename + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))

    def _generate_coco_for_image(self, filename, width, height, node_bboxes, edge_mask):

        self.coco["images"].append({
            "id": self.img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        for min_x, min_y, max_x, max_y in node_bboxes:
            w = max_x - min_x
            h = max_y - min_y
            x = min_x
            y = min_y

            node_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(node_mask, (int(x + w / 2), int(y + h / 2)), (int(w / 2), int(h / 2)), 0, 0, 360, 255, -1)
            polys = extract_contours(node_mask)

            for poly in polys:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                self.coco["annotations"].append({
                    "id": self.ann_id,
                    "image_id": self.img_id,
                    "category_id": 1,
                    "segmentation": [[int(c) for pt in poly for c in pt]],
                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
                    "area": float(polygon_area(xs, ys)),
                    "iscrowd": 0
                })
                self.ann_id += 1

        polys = extract_contours(edge_mask)
        for poly in polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            self.coco["annotations"].append({
                "id": self.ann_id,
                "image_id": self.img_id,
                "category_id": 2,
                "segmentation": [[int(c) for pt in poly for c in pt]],
                "bbox": [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
                "area": float(polygon_area(xs, ys)),
                "iscrowd": 0
            })
            self.ann_id += 1

        self.img_id += 1

    def generate_annotations(self, filename, width, height, node_bboxes, edge_bboxes, edge_mask):
        if self.generate_yolo:
            self._generate_yolo_for_image(filename, width, height, node_bboxes, edge_bboxes)
        else:
            self._generate_coco_for_image(filename, width, height, node_bboxes, edge_mask)

    def finalize(self, image_output_folder):
        if not self.generate_yolo:
            with open(self.coco_output_file, "w") as f:
                json.dump(self.coco, f)
            print(os.path.abspath(self.coco_output_file))
        else:
            print(os.path.abspath(self.yolo_label_folder))

        print(os.path.abspath(image_output_folder))


def main():
    if GENERATE_YOLO_LABELS:
        IMAGE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "images")
    else:
        IMAGE_OUTPUT_FOLDER = OUTPUT_FOLDER

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

    pygame.init()

    try:
        data = json.load(open(INPUT_JSON))
    except FileNotFoundError:
        return

    generator = AnnotationGenerator(GENERATE_YOLO_LABELS, OUTPUT_FOLDER)

    for graph in data:
        image_node_color = get_random_color()
        image_edge_color = get_random_color()

        vertices = graph["vertices"]
        edges = graph["connections"]
        width = int(graph.get("img_width", 800))
        height = int(graph.get("img_height", 800))
        img_filename = graph["filename"]

        image_path = os.path.join(IMAGE_OUTPUT_FOLDER, img_filename)

        drawer = GraphDrawer(width, height, vertices, edges, image_node_color, image_edge_color)
        node_bboxes, edge_bboxes, edge_mask = drawer.draw(image_path)

        generator.generate_annotations(img_filename, width, height, node_bboxes, edge_bboxes, edge_mask)


    generator.finalize(IMAGE_OUTPUT_FOLDER)
    pygame.quit()

if __name__ == "__main__":
    main()