import pygame
import json
import os
import math

with open("dbg.json") as f:
    data = json.load(f)

OUTPUT_FOLDER = "dbg"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BG_COLOR     = (255, 250, 240)
NODE_COLOR   = (255, 140, 66)
NODE_BORDER  = (210, 90, 45)
EDGE_COLOR   = (180, 80, 60)
COST_COLOR   = (120, 60, 40)

DBG_EDGE     = (255, 70, 70)
DBG_VERT     = (70, 130, 180)

DBG = True

pygame.init()
clock = pygame.time.Clock()


def render_graph(graph):
    vertices = graph["vertices"]
    edges = graph["connections"]

    SCREEN_WIDTH = int(graph.get("img_width", 800))
    SCREEN_HEIGHT = int(graph.get("img_height", 800))
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(BG_COLOR)

    # First pass: draw all edges
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

        if DBG:
            edge_rect = pygame.Rect(
                min(start_pos[0], end_pos[0]),
                min(start_pos[1], end_pos[1]),
                max(abs(w), 1),
                max(abs(h), 1)
            )
            pygame.draw.rect(screen, DBG_EDGE, edge_rect, 1)

    # Second pass: draw cost labels with smart positioning
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

        cost = e.get("cost", "")
        if cost:
            font = pygame.font.SysFont("Arial", 22)
            cost_text = font.render(str(cost), True, COST_COLOR)
            text_rect = cost_text.get_rect()

            # Calculate midpoint
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2

            # Calculate line angle to determine positioning
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            angle = math.degrees(math.atan2(dy, dx))

            # Adjust position based on line orientation
            offset_distance = 15  # Distance from the line

            if -45 <= angle <= 45:
                # Mostly horizontal line - place above or below
                if dy >= 0:
                    # Line goes down-right, place above
                    text_pos = (mid_x, mid_y - offset_distance)
                else:
                    # Line goes up-right, place below
                    text_pos = (mid_x, mid_y + offset_distance)
            elif 45 < angle <= 135:
                # Mostly vertical line going down - place to the left
                text_pos = (mid_x - offset_distance, mid_y)
            elif -135 <= angle < -45:
                # Mostly vertical line going up - place to the right
                text_pos = (mid_x + offset_distance, mid_y)
            else:
                # Other cases - place to the right
                text_pos = (mid_x + offset_distance, mid_y)

            text_rect.center = text_pos
            screen.blit(cost_text, text_rect)

    # Draw vertices
    for v in vertices:
        x = v["x"] * SCREEN_WIDTH
        y = v["y"] * SCREEN_HEIGHT
        w = v["width"] * SCREEN_WIDTH
        h = v["height"] * SCREEN_HEIGHT

        node_rect = pygame.Rect(x, y, w, h)
        pygame.draw.ellipse(screen, NODE_COLOR, node_rect)
        pygame.draw.ellipse(screen, NODE_BORDER, node_rect, 2)

        if DBG:
            pygame.draw.rect(screen, DBG_VERT, node_rect, 1)

        font = pygame.font.SysFont("Arial", max(12, int(h * 0.6 / max(1, math.log(len(v["symbol"]) + 1, 3)))))
        text = font.render(v["symbol"], True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + w / 2, y + h / 2))
        screen.blit(text, text_rect)

    filename = os.path.join(OUTPUT_FOLDER, graph["filename"])
    pygame.image.save(screen, filename)
    print(f"✅ Saved: {filename}")


for i, graph in enumerate(data):
    render_graph(graph)
    clock.tick(10)

pygame.quit()
print(f"🎉 All graphs saved in the '{OUTPUT_FOLDER}' folder!")