import pygame
import sys
import math
import json

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
NODE_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (200, 150, 100), (150, 200, 100), (100, 150, 200),
    (200, 100, 150), (150, 100, 200)
]

class GraphRenderer:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("GRAPH VISUALIZER - RUST OUTPUT")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Graph data
        self.vertices = []
        self.edges = []
        self.img_width = 0
        self.img_height = 0

        # Visualization state
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.selected_node = None
        self.show_labels = True

    def load_data(self, json_data):
        """Load the fucking JSON data"""
        self.img_width = json_data.get("img_width", 800)
        self.img_height = json_data.get("img_height", 600)

        # Load vertices
        self.vertices = []
        for v_data in json_data.get("vertices", []):
            vertex = {
                'id': v_data.get('id', 0),
                'symbol': str(v_data.get('symbol', 'A')),
                'x': v_data.get('x', 0) * self.img_width,
                'y': v_data.get('y', 0) * self.img_height,
                'width': v_data.get('width', 40) * self.img_width,
                'height': v_data.get('height', 40) * self.img_height,
                'neighbours': v_data.get('neighbours', [])
            }
            self.vertices.append(vertex)

        # Load edges
        self.edges = []
        for e_data in json_data.get("connections", []):
            edge = {
                'id': e_data.get('id', 0),
                'relationship': e_data.get('relationship', [0, 0]),
                'x': e_data.get('x', 0) * self.img_width,
                'y': e_data.get('y', 0) * self.img_height,
                'width': e_data.get('width', 0) * self.img_width,
                'height': e_data.get('height', 0) * self.img_height,
                'logic_label': e_data.get('logic_label', 'edge')
            }
            self.edges.append(edge)

    def transform_coords(self, x, y):
        """Transform coordinates based on zoom and pan"""
        return (x * self.zoom + self.offset_x, y * self.zoom + self.offset_y)

    def draw_graph(self):
        """Draw the entire fucking graph"""
        self.screen.fill(WHITE)

        # Draw edges first (so they're behind nodes)
        for edge in self.edges:
            self.draw_edge(edge)

        # Draw nodes
        for vertex in self.vertices:
            self.draw_vertex(vertex)

        # Draw info panel
        self.draw_info_panel()

    def draw_edge(self, edge):
        """Draw a connection between nodes"""
        rel = edge['relationship']
        if len(rel) < 2:
            return

        # Find the actual vertices for this edge
        v1 = next((v for v in self.vertices if v['id'] == rel[0]), None)
        v2 = next((v for v in self.vertices if v['id'] == rel[1]), None)

        if not v1 or not v2:
            return

        # Calculate center points
        x1 = v1['x'] + v1['width'] / 2
        y1 = v1['y'] + v1['height'] / 2
        x2 = v2['x'] + v2['width'] / 2
        y2 = v2['y'] + v2['height'] / 2

        # Transform coordinates
        tx1, ty1 = self.transform_coords(x1, y1)
        tx2, ty2 = self.transform_coords(x2, y2)

        # Draw line
        pygame.draw.line(self.screen, DARK_GRAY, (tx1, ty1), (tx2, ty2), 2)

        # Draw edge ID near the middle of the line
        mid_x = (tx1 + tx2) / 2
        mid_y = (ty1 + ty2) / 2
        edge_text = self.small_font.render(f"E{edge['id']}", True, BLACK)
        self.screen.blit(edge_text, (mid_x + 5, mid_y + 5))

    def draw_vertex(self, vertex):
        """Draw a node"""
        # Transform coordinates
        x, y = self.transform_coords(vertex['x'], vertex['y'])
        width = vertex['width'] * self.zoom
        height = vertex['height'] * self.zoom

        # Choose color based on node ID
        color_idx = vertex['id'] % len(NODE_COLORS)
        color = NODE_COLORS[color_idx]

        # Draw node
        pygame.draw.ellipse(self.screen, color, (x, y, width, height))
        pygame.draw.ellipse(self.screen, BLACK, (x, y, width, height), 2)

        # Draw node label
        if self.show_labels:
            label = f"{vertex['symbol']}({vertex['id']})"
            text = self.font.render(label, True, BLACK)
            text_rect = text.get_rect(center=(x + width/2, y + height/2))
            self.screen.blit(text, text_rect)

        # Highlight selected node
        if self.selected_node == vertex['id']:
            pygame.draw.ellipse(self.screen, RED, (x-5, y-5, width+10, height+10), 3)

    def draw_info_panel(self):
        """Draw information panel"""
        panel_rect = pygame.Rect(10, 10, 300, 120)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)

        # Display info
        info_lines = [
            f"Nodes: {len(self.vertices)}",
            f"Edges: {len(self.edges)}",
            f"Zoom: {self.zoom:.2f}x",
            "Controls:",
            "Mouse: Pan & Select",
            "Wheel: Zoom",
            "L: Toggle Labels"
        ]

        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, BLACK)
            self.screen.blit(text, (20, 20 + i * 16))

        # Show selected node info
        if self.selected_node is not None:
            node = next((v for v in self.vertices if v['id'] == self.selected_node), None)
            if node:
                selected_text = [
                    f"Selected: {node['symbol']}({node['id']})",
                    f"Neighbors: {len(node['neighbours'])}"
                ]
                for i, line in enumerate(selected_text):
                    text = self.small_font.render(line, True, BLUE)
                    self.screen.blit(text, (20, 140 + i * 16))

    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                    # Check for node selection
                    mouse_pos = pygame.mouse.get_pos()
                    self.check_node_selection(mouse_pos)

                elif event.button == 4:  # Scroll up - zoom in
                    self.zoom *= 1.1

                elif event.button == 5:  # Scroll down - zoom out
                    self.zoom /= 1.1
                    self.zoom = max(0.1, self.zoom)  # Minimum zoom

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click release
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.offset_x += dx
                    self.offset_y += dy
                    self.last_mouse_pos = event.pos

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:  # Toggle labels
                    self.show_labels = not self.show_labels
                elif event.key == pygame.K_r:  # Reset view
                    self.offset_x = 0
                    self.offset_y = 0
                    self.zoom = 1.0
                    self.selected_node = None

        return True

    def check_node_selection(self, mouse_pos):
        """Check if a node was clicked"""
        for vertex in self.vertices:
            # Transform node coordinates for hit testing
            x, y = self.transform_coords(vertex['x'], vertex['y'])
            width = vertex['width'] * self.zoom
            height = vertex['height'] * self.zoom

            # Create ellipse hitbox
            center_x = x + width / 2
            center_y = y + height / 2
            dx = (mouse_pos[0] - center_x) / (width / 2)
            dy = (mouse_pos[1] - center_y) / (height / 2)

            # Check if click is inside ellipse
            if dx * dx + dy * dy <= 1:
                self.selected_node = vertex['id']
                return

        # Clicked on empty space - deselect
        self.selected_node = None

    def run(self):
        """Main fucking loop"""
        running = True
        while running:
            running = self.handle_events()
            self.draw_graph()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    with open("dataset/synth_graphs.json") as f:
        json_data = json.load(f)[100]

    renderer = GraphRenderer()
    renderer.load_data(json_data)  # This should be your complete JSON object
    renderer.run()

pygame.quit()
sys.exit()