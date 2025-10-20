import pygame
import json
import os
import math

# === Load all graph data ===
with open("dbg.json") as f:
    data = json.load(f)

# === Config ===
OUTPUT_FOLDER = "dbg"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Warm Color Palette ===
BG_COLOR     = (255, 250, 240)   # soft warm cream background
NODE_COLOR   = (255, 140, 66)    # rich warm orange
NODE_BORDER  = (210, 90, 45)     # deeper burnt orange border
EDGE_COLOR   = (180, 80, 60)     # warm brown-red edges

# === Debug Colors ===
DBG_EDGE     = (255, 70, 70)     # bright red for edge bounds
DBG_VERT     = (70, 130, 180)    # steel blue for vertex bounds

DBG = True  # 🔧 Set to True for debug bounding boxes

pygame.init()
clock = pygame.time.Clock()


def render_graph(graph):
    vertices = graph["vertices"]
    edges = graph["connections"]

    SCREEN_WIDTH = int(graph.get("img_width", 800))
    SCREEN_HEIGHT = int(graph.get("img_height", 800))
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(BG_COLOR)

    # Draw edges
    for e in edges:
        x = e["x"] * SCREEN_WIDTH
        y = e["y"] * SCREEN_HEIGHT
        w = e["width"] * SCREEN_WIDTH
        h = e["height"] * SCREEN_HEIGHT
        v1, v2 = e["relationship"]

        from_top = (
            vertices[v1]["y"] < vertices[v2]["y"]
            if vertices[v1]["x"] <= vertices[v2]["x"]
            else vertices[v1]["y"] > vertices[v2]["y"]
        )

        if from_top:
            start_pos = (x, y)
            end_pos = (x + w, y + h)
        else:
            start_pos = (x, y + h)
            end_pos = (x + w, y)

        pygame.draw.line(screen, EDGE_COLOR, start_pos, end_pos, 4)

        # 🔵 Draw edge debug rect
        if DBG:
            edge_rect = pygame.Rect(
                min(start_pos[0], end_pos[0]),
                min(start_pos[1], end_pos[1]),
                max(abs(w), 1),
                max(abs(h), 1)
            )
            pygame.draw.rect(screen, DBG_EDGE, edge_rect, 1)

    # Draw vertices
    for v in vertices:
        x = v["x"] * SCREEN_WIDTH
        y = v["y"] * SCREEN_HEIGHT
        w = v["width"] * SCREEN_WIDTH
        h = v["height"] * SCREEN_HEIGHT

        node_rect = pygame.Rect(x, y, w, h)
        pygame.draw.ellipse(screen, NODE_COLOR, node_rect)
        pygame.draw.ellipse(screen, NODE_BORDER, node_rect, 2)

        # 🟥 Draw node debug rect
        if DBG:
            pygame.draw.rect(screen, DBG_VERT, node_rect, 1)

        # Symbol text
        font = pygame.font.SysFont("Arial", max(12, int(h * 0.6 / max(1, math.log(len(v["symbol"]) + 1, 3)))))
        text = font.render(v["symbol"], True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + w / 2, y + h / 2))
        screen.blit(text, text_rect)

    # Save rendered image
    filename = os.path.join(OUTPUT_FOLDER, graph["filename"])
    pygame.image.save(screen, filename)
    print(f"✅ Saved: {filename}")


# === Render all graphs ===
for i, graph in enumerate(data):
    render_graph(graph)
    clock.tick(10)  # small delay between renders

pygame.quit()
print(f"🎉 All graphs saved in the '{OUTPUT_FOLDER}' folder!")
