import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Ellipse, Polygon


COLORS = plt.cm.tab20(np.linspace(0, 1, 100))


def _face_color(idx):
    return COLORS[idx % len(COLORS)]
def plot_labeled_faces(labeled_clean, save_path="labeled.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(labeled_clean, cmap='tab20')
    plt.colorbar()
    plt.title(f"Labeled regions: {labeled_clean.max()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_labeled_faces_with_num(labeled_clean, save_path="labeled.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(labeled_clean, cmap='tab20')
    plt.colorbar()

    for label_id in range(1, labeled_clean.max() + 1):
        mask = labeled_clean == label_id
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()
        plt.text(cx, cy, str(label_id), color='white', fontsize=4)

    plt.title(f"Labeled regions: {labeled_clean.max()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_corners(img, face_corners, face_pixels, valid_labels, save_path="corners.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img)
    axes[0].set_title("Detected corners per face")

    axes[1].set_facecolor('black')
    axes[1].set_xlim(0, img.shape[1])
    axes[1].set_ylim(img.shape[0], 0)
    axes[1].set_title("Corner polygons only")

    for idx, (face_id, corners) in enumerate(face_corners.items()):
        color = _face_color(idx)
        centroid = face_pixels[face_id].mean(axis=0)

        # Left panel
        axes[0].scatter(corners[:, 1], corners[:, 0], c=[color], s=1, zorder=5)
        axes[0].add_patch(Polygon(corners[:, ::-1], fill=False, edgecolor=color, linewidth=.5, ))
        # axes[0].text(centroid[1], centroid[0], str(face_id),
        #              fontsize=8, color='white', ha='center', va='center', fontweight='bold')

        # Right panel
        axes[1].add_patch(Polygon(corners[:, ::-1], fill=True,
                                  facecolor=color, edgecolor='white', linewidth=2, alpha=0.7))
        axes[1].scatter(corners[:, 1], corners[:, 0], c='white', s=50, zorder=5)
        # axes[1].text(centroid[1], centroid[0], str(face_id),
        #              fontsize=8, color='white', ha='center', va='center', fontweight='bold')
    print("plotted corners")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_adjacency(img, labeled_clean, valid_labels,
                   face_centroids, adjacent_faces, save_path="adjacency.png"):
    G = nx.Graph()
    for i in valid_labels:
        G.add_node(i)
    for i in valid_labels:
        for j in adjacent_faces[i]:
            if j > i:
                G.add_edge(i, j)

    fig, axes = plt.subplots(1, 1, figsize=(18, 8))

    axes.imshow(img)
    axes.set_title(f"Face adjacency graph: {G.number_of_edges()} edges")
    nx.draw(G, pos={i: (face_centroids[i][1], face_centroids[i][0]) for i in valid_labels},
            ax=axes, with_labels=True, node_color='yellow',
            edge_color='red', node_size=1, font_size=8, width=1)


    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_merged_vertices(img, face_corners, all_pts, vertices,
                         valid_labels, adjacency, face_pt_indices, save_path="merges.png"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img)
    ax.set_title(f"Merged vertices: {len(vertices)}")

    # Draw face polygons
    for idx, (face_id, corners) in enumerate(face_corners.items()):
        ax.add_patch(Polygon(corners[:, ::-1], fill=True,
                             facecolor=_face_color(idx), edgecolor='white',
                             linewidth=1, alpha=0.3))

    # Final merged vertices in red
    for vid, pos in vertices.items():
        ax.scatter(pos[1], pos[0], c='red', s=2, zorder=6)
        ax.text(pos[1] + 5, pos[0] - 5, str(vid), fontsize=7, color='white', zorder=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_connections(img, ellipses, connections, save_path="connections.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray')

    for id_a, id_b, angle_diff in connections:
        ca = ellipses[id_a]['center']
        cb = ellipses[id_b]['center']
        ax.plot([ca[0], cb[0]], [ca[1], cb[1]], 'c-', linewidth=1, alpha=0.7)

    for label_id, ell in ellipses.items():
        cx, cy = ell['center']
        ax.plot(cx, cy, 'r.', markersize=5)
        ax.text(cx, cy - 5, str(label_id), color='yellow', fontsize=7,
                ha='center', fontweight='bold')

    ax.set_title(f"Aligned connections: {len(connections)}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_combined_visualization(img, face_corners, vertices, valid_labels, 
                                face_centroids, adjacent_faces, save_path="combined_viz.png"):
    """
    Plots the adjacency graph overlaid on the image with merged vertices 
    and face polygons.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.imshow(img)
    
    # 1. Draw Face Polygons (Base layer)
    # Note: Assumes _face_color is defined in your existing module
    for idx, (face_id, corners) in enumerate(face_corners.items()):
        ax.add_patch(Polygon(corners[:, ::-1], fill=True, 
                             facecolor=_face_color(idx), edgecolor='white', 
                             linewidth=1, alpha=0.3))

    # 2. Draw Adjacency Graph (Middle layer)
    G = nx.Graph()
    for i in valid_labels:
        G.add_node(i)
        for j in adjacent_faces.get(i, []):
            if j > i:
                G.add_edge(i, j)

    pos_dict = {i: (face_centroids[i][1], face_centroids[i][0]) for i in valid_labels}
    
    nx.draw(G, pos=pos_dict, ax=ax, with_labels=True, node_color='yellow',
            edge_color='red', node_size=5, font_size=8, width=1)

    # 3. Draw Merged Vertices (Top layer)
    for vid, pos in vertices.items():
        ax.scatter(pos[1], pos[0], c='red', s=5, zorder=6)
        ax.text(pos[1] + 5, pos[0] - 5, str(vid), fontsize=7, color='white', zorder=7)

    ax.set_title(f"Combined Visualization | Edges: {G.number_of_edges()} | Vertices: {len(vertices)}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()