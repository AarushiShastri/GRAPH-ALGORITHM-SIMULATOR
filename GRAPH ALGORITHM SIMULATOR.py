import networkx as nx
import heapq
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import random

# Define pastel color palette
PASTEL_COLORS = {
    'nodes': ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D4BAFF', '#F3BAFF'],  # Pastel VIBGYOR
    'edges': '#FFC300',
    'highlighted_edge': '#FF5733',
    'visited_node': '#900C3F',
    'processing_node': '#DAF7A6',
    'comparison_node': '#FFBF00',
    'final_node': '#3498DB',
    'default_node': '#008080',
    'default_edge': '#2C3E50',
    'text': '#4A4A4A',
    'background': '#fbf2c4'
}

class GraphAlgorithmApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Algorithm Simulator")
        self.geometry("600x400")
        self.configure(bg=PASTEL_COLORS['background'])

        self.algorithm_type = None

        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(self, text="Choose Algorithm Type", font=("Helvetica", 18),
                         bg=PASTEL_COLORS['background'], fg=PASTEL_COLORS['text'])
        title.pack(pady=20)

        traversal_btn = tk.Button(self, text="Graph Traversal Algorithms", font=("Helvetica", 14),
                                  bg=PASTEL_COLORS['nodes'][3], fg=PASTEL_COLORS['text'],
                                  command=self.select_traversal)
        traversal_btn.pack(pady=10)

        mst_btn = tk.Button(self, text="Minimum Spanning Tree Algorithms", font=("Helvetica", 14),
                            bg=PASTEL_COLORS['nodes'][5], fg=PASTEL_COLORS['text'],
                            command=self.select_mst)
        mst_btn.pack(pady=10)

    def select_traversal(self):
        self.algorithm_type = 'traversal'
        self.open_graph_creation_window()

    def select_mst(self):
        self.algorithm_type = 'mst'
        self.open_graph_creation_window()

    def open_graph_creation_window(self):
        self.withdraw()  # Hide the main window
        graph_creation_window = tk.Toplevel(self)
        graph_creation_window.geometry("800x600")
        graph_creation_window.configure(bg=PASTEL_COLORS['background'])
        GraphCreationWindow(graph_creation_window, self.algorithm_type, self)


class GraphCreationWindow:
    def __init__(self, master, algorithm_type, main_app):
        self.algorithm_type = algorithm_type
        self.master = master
        self.main_app = main_app
        self.G = nx.Graph()
        self.node_positions = {}
        self.node_labels = {}
        self.node_count = 0
        self.selected_nodes = []
        self.current_mode = 'create_nodes'

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title("Graph Creation", fontsize=16)
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        self.create_mode_buttons()

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def create_mode_buttons(self):
        mode_frame = tk.Frame(self.master, bg=PASTEL_COLORS['background'])
        mode_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_nodes_btn = tk.Button(mode_frame, text="Create Nodes", bg=PASTEL_COLORS['nodes'][0],
                                          fg=PASTEL_COLORS['text'], command=self.set_create_nodes_mode)
        self.create_nodes_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.create_edges_btn = tk.Button(mode_frame, text="Create Edges", bg=PASTEL_COLORS['nodes'][1],
                                          fg=PASTEL_COLORS['text'], command=self.set_create_edges_mode)
        self.create_edges_btn.pack(side=tk.LEFT, padx=10, pady=10)

        run_simulation_btn = tk.Button(mode_frame, text="Run Simulation", bg=PASTEL_COLORS['highlighted_edge'],
                                       fg=PASTEL_COLORS['text'], command=self.run_simulation)
        run_simulation_btn.pack(side=tk.LEFT, padx=10, pady=10)

        randomize_weights_btn = tk.Button(mode_frame, text="Randomize Weights", bg=PASTEL_COLORS['nodes'][2],
                                          fg=PASTEL_COLORS['text'], command=self.randomize_weights)
        randomize_weights_btn.pack(side=tk.LEFT, padx=10, pady=10)

        clear_canvas_btn = tk.Button(mode_frame, text="Clear Canvas", bg=PASTEL_COLORS['nodes'][6],
                                     fg=PASTEL_COLORS['text'], command=self.clear_canvas)
        clear_canvas_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.highlight_current_mode()

    def clear_canvas(self):
        self.G.clear()
        self.node_positions.clear()
        self.node_labels.clear()
        self.node_count = 0
        self.selected_nodes = []
        self.draw_graph()

    def set_create_nodes_mode(self):
        self.current_mode = 'create_nodes'
        self.highlight_current_mode()

    def set_create_edges_mode(self):
        self.current_mode = 'create_edges'
        self.highlight_current_mode()

    def highlight_current_mode(self):
        buttons = [self.create_nodes_btn, self.create_edges_btn]
        for btn in buttons:
            if btn.cget("text").startswith(self.current_mode.capitalize()):
                btn.config(bg=PASTEL_COLORS['highlighted_edge'], fg=PASTEL_COLORS['text'])
            else:
                btn.config(bg=PASTEL_COLORS['nodes'][1], fg=PASTEL_COLORS['text'])

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if self.current_mode == 'create_nodes':
            pos = (event.xdata, event.ydata)
            if pos[0] is None or pos[1] is None:
                return
            node_id = chr(97 + self.node_count)
            self.node_count += 1
            self.node_positions[node_id] = pos
            self.node_labels[node_id] = node_id
            self.G.add_node(node_id, pos=pos)
            self.draw_graph()

        elif self.current_mode == 'create_edges':
            pos = (event.xdata, event.ydata)
            if pos[0] is None or pos[1] is None:
                return
            node_id = self.get_closest_node(pos)
            if node_id is None:
                return
            if len(self.selected_nodes) == 0:
                if node_id:
                    self.selected_nodes.append(node_id)
                    self.highlight_node(node_id, PASTEL_COLORS['visited_node'])
            elif len(self.selected_nodes) == 1:
                if node_id and node_id != self.selected_nodes[0]:
                    self.G.add_edge(self.selected_nodes[0], node_id,
                                    weight=random.randint(1, 20))  # Random positive weight
                    self.draw_graph()
                    self.selected_nodes = []
            else:
                self.selected_nodes = []

    def get_closest_node(self, pos):
        """Finds the node closest to the clicked position with a reasonable threshold."""
        closest_node = None
        min_distance = float('inf')
        threshold_distance = 0.1  # Adjusted threshold to ensure precision in node selection
        for node, node_pos in self.node_positions.items():
            distance = ((node_pos[0] - pos[0]) ** 2 + (node_pos[1] - pos[1]) ** 2) ** 0.5
            if distance < threshold_distance and distance < min_distance:
                min_distance = distance
                closest_node = node
        return closest_node

    def highlight_node(self, node_id, color):
        self.draw_graph()
        nx.draw_networkx_nodes(self.G, pos=self.node_positions, nodelist=[node_id],
                               node_color=color, node_size=700, ax=self.ax)
        self.canvas.draw()

    def draw_graph(self):
        self.ax.clear()
        nx.draw(self.G, pos=self.node_positions, ax=self.ax, with_labels=True, labels=self.node_labels,
                node_color=PASTEL_COLORS['default_node'], node_size=500, font_color="#fbf2c4",
                font_weight="bold", edge_color=PASTEL_COLORS['default_edge'])
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos=self.node_positions, edge_labels=edge_labels,
                                     ax=self.ax, font_color=PASTEL_COLORS['text'])
        self.canvas.draw()

    def randomize_weights(self):
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = random.randint(1, 20)  # Random positive weight
        self.draw_graph()

    def run_simulation(self):
        if len(self.G.nodes) == 0:
            messagebox.showerror("Error", "Please create a graph before running the simulation.")
            return

        start_node = simpledialog.askstring("Input", "Enter the start node:", parent=self.master)
        if not start_node or start_node not in self.G.nodes:
            messagebox.showerror("Error", "Invalid start node.")
            return

        SimulationWindow(self.master, self.G, self.algorithm_type, start_node, self.main_app)


class SimulationWindow:
    def __init__(self, master, G, algorithm_type, start_node, main_app):
        self.master = master
        self.G = G
        self.algorithm_type = algorithm_type
        self.start_node = start_node
        self.main_app = main_app

        self.window = tk.Toplevel(master)
        self.window.geometry("1200x700")
        self.window.title("Simulation")
        self.window.configure(bg=PASTEL_COLORS['background'])

        # Top frame for Back button
        top_frame = tk.Frame(self.window, bg=PASTEL_COLORS['background'])
        top_frame.pack(side=tk.TOP, fill=tk.X)

        back_btn = tk.Button(top_frame, text="Back to Main Menu", bg=PASTEL_COLORS['nodes'][0],
                             fg=PASTEL_COLORS['text'], command=self.back_to_main_menu)
        back_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        self.animation_interval = 1500  # 1.5 seconds per step

        # Determine number of algorithms to display
        if self.algorithm_type == 'traversal':
            self.num_algorithms = 2
        elif self.algorithm_type == 'mst':
            self.num_algorithms = 2

        if self.num_algorithms == 2:
            self.left_frame = tk.Frame(self.window, width=600, bg=PASTEL_COLORS['background'])
            self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.right_frame = tk.Frame(self.window, width=600, bg=PASTEL_COLORS['background'])
            self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            self.fig1, self.ax1 = plt.subplots(figsize=(5, 4))
            self.fig2, self.ax2 = plt.subplots(figsize=(5, 4))

            self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.left_frame)
            self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.right_frame)

            self.canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.label1 = tk.Label(self.left_frame, text="", font=("Helvetica", 10),
                                   bg=PASTEL_COLORS['background'], fg=PASTEL_COLORS['text'])
            self.label1.pack(pady=10)
            self.label2 = tk.Label(self.right_frame, text="", font=("Helvetica", 10),
                                   bg=PASTEL_COLORS['background'], fg=PASTEL_COLORS['text'])
            self.label2.pack(pady=10)

        self.run_simulation()

    def back_to_main_menu(self):
        self.window.destroy()
        self.master.destroy()
        self.main_app.deiconify()

    def run_simulation(self):
        if self.algorithm_type == 'traversal':
            threading.Thread(target=self.animate_algorithm,
                             args=('BFS', self.bfs, self.ax1, self.canvas1, self.label1)).start()
            threading.Thread(target=self.animate_algorithm,
                             args=('DFS', self.dfs, self.ax2, self.canvas2, self.label2)).start()
        elif self.algorithm_type == 'mst':
            threading.Thread(target=self.animate_algorithm,
                             args=('Prim\'s', self.prims, self.ax1, self.canvas1, self.label1)).start()
            threading.Thread(target=self.animate_algorithm,
                             args=('Kruskal\'s', self.kruskals, self.ax2, self.canvas2, self.label2)).start()

    def animate_algorithm(self, algorithm_name, algorithm_function, ax, canvas, label):
        animation_step = 0
        result = algorithm_function()

        if result is None:
            return

        path_info, additional_info = result
        path, nodes_visited = path_info

        num_steps = len(path) if path else 1

        def update_canvas():
            nonlocal animation_step
            if animation_step < num_steps:
                ax.clear()
                self.draw_graph(ax, path[:animation_step+1], animation_step)
                info_text = f"{algorithm_name}\n"
                info_text += f"Starting Node: {self.start_node}\n"

                # Display relevant data structures per algorithm
                if algorithm_name == 'BFS':
                    queue_snapshot = additional_info['queue_snapshots'][animation_step]
                    info_text += f"Queue (FIFO): {queue_snapshot}\n"
                elif algorithm_name == 'DFS':
                    stack_snapshot = additional_info['stack_snapshots'][animation_step]
                    info_text += f"Stack (LIFO): {stack_snapshot}\n"
                elif algorithm_name == "Kruskal's":
                    sorted_edges = additional_info['sorted_edges']
                    info_text += f"Sorted Edges: {sorted_edges}\n"
                elif algorithm_name == "Prim's":
                    info_text += f"Current Node: {nodes_visited[animation_step]}\n"

                info_text += f"Visited Nodes: {', '.join(map(str, nodes_visited[:animation_step+1]))}\n"
                if 'time_complexity' in additional_info:
                    info_text += f"Time Complexity: {additional_info['time_complexity']}\n"

                if path:
                    shortest_path_text = ' -> '.join(map(str, [u for u, v in path[:animation_step+1]] + [path[animation_step][1]]))
                    info_text += f"Path: {shortest_path_text}\n"

                label.config(text=info_text)
                canvas.draw()

                animation_step += 1
                self.master.after(self.animation_interval, update_canvas)

        update_canvas()

    def draw_graph(self, ax, path, step):
        pos = nx.spring_layout(self.G, seed=42)
        ax.clear()

        # Draw the complete graph
        nx.draw(self.G, pos=pos, ax=ax, with_labels=True,
                node_color=PASTEL_COLORS['default_node'], node_size=400,
                font_color="#fbf2c4", font_weight="bold", edge_color=PASTEL_COLORS['default_edge'])

        # Highlight edges and nodes in the path
        if path:
            nx.draw_networkx_edges(self.G, pos=pos, edgelist=path, edge_color=PASTEL_COLORS['highlighted_edge'],
                                   width=2, ax=ax)

        nodes_in_path = set(n for u, v in path for n in (u, v))

        # Then mark the nodes that are finalized
        nx.draw_networkx_nodes(self.G, pos=pos, nodelist=nodes_in_path,
                               node_color=PASTEL_COLORS['final_node'], node_size=500, ax=ax)

        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=edge_labels,
                                     ax=ax, font_color=PASTEL_COLORS['text'])

        ax.figure.canvas.draw_idle()

    def bfs(self):
        path_edges = []
        visited = []
        queue = [self.start_node]
        visited_set = set()
        queue_snapshots = []

        while queue:
            node = queue.pop(0)
            if node in visited_set:
                continue
            visited_set.add(node)
            visited.append(node)

            neighbors = [n for n in self.G.neighbors(node) if n not in visited_set and n not in queue]
            queue.extend(neighbors)
            for neighbor in neighbors:
                path_edges.append((node, neighbor))

            queue_snapshots.append(list(queue))

        additional_info = {
            'queue_snapshots': queue_snapshots,
            'time_complexity': 'O(V + E)',
        }
        return (path_edges, visited), additional_info

    def dfs(self):
        path_edges = []
        visited = []
        stack = [self.start_node]
        visited_set = set()
        stack_snapshots = []

        while stack:
            node = stack.pop()
            if node in visited_set:
                continue
            visited_set.add(node)
            visited.append(node)

            neighbors = [n for n in self.G.neighbors(node) if n not in visited_set and n not in stack]
            stack.extend(neighbors)
            for neighbor in neighbors:
                path_edges.append((node, neighbor))

            stack_snapshots.append(list(stack))

        additional_info = {
            'stack_snapshots': stack_snapshots,
            'time_complexity': 'O(V + E)',
        }
        return (path_edges, visited), additional_info

    def prims(self):
        try:
            mst, nodes_visited = self._prims_algorithm()
            path_edges = list(mst.edges)
            additional_info = {
                'time_complexity': 'O(E log V)',
            }
            return (path_edges, nodes_visited), additional_info
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            return None

    def _prims_algorithm(self):
        mst = nx.Graph()
        visited = set()
        edges = [(0, self.start_node, None)]  # (weight, node, from_node)
        nodes_visited = []

        while edges:
            weight, node, from_node = heapq.heappop(edges)

            if node in visited:
                continue

            visited.add(node)
            nodes_visited.append(node)
            if from_node is not None:
                mst.add_edge(node, from_node, weight=weight)

            for neighbor, data in self.G[node].items():
                if neighbor not in visited:
                    heapq.heappush(edges, (data.get('weight', 1), neighbor, node))

        return mst, nodes_visited

    def kruskals(self):
        try:
            mst, nodes_visited, sorted_edges = self._kruskals_algorithm()
            path_edges = list(mst.edges)
            additional_info = {
                'sorted_edges': sorted_edges,
                'time_complexity': 'O(E log E)',
            }
            return (path_edges, nodes_visited), additional_info
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            return None

    def _kruskals_algorithm(self):
        mst = nx.Graph()
        edges = sorted(self.G.edges(data=True), key=lambda x: x[2].get('weight', 1))
        sorted_edges = [(u, v, data.get('weight', 1)) for u, v, data in edges]
        parent = {}
        rank = {}
        nodes_visited = []

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                else:
                    parent[root1] = root2
                    if rank[root1] == rank[root2]:
                        rank[root2] += 1

        for node in self.G.nodes:
            parent[node] = node
            rank[node] = 0

        for u, v, data in edges:
            if find(u) != find(v):
                mst.add_edge(u, v, weight=data.get('weight', 1))
                union(u, v)
                nodes_visited.extend([u, v])

        return mst, nodes_visited, sorted_edges


if __name__ == "__main__":
    app = GraphAlgorithmApp()
    app.mainloop()