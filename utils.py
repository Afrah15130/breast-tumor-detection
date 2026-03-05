# utils.py
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
import os

def safe_save_image(img, path):
    """Save image safely in uint8 format."""
    if img.dtype != np.uint8:
        img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def crop_two_breasts(img):
    """Crop only frontal breast area, remove neck/sides/lower body."""
    h, w, _ = img.shape
    top = int(h * 0.15)
    bottom = int(h * 0.85)
    left = int(w * 0.2)
    right = int(w * 0.8)
    return img[top:bottom, left:right]

def superpixel_segmentation(img, n_segments=200):
    """Return superpixel segmented image."""
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=1)
    superpixel_img = label2rgb(segments, img, kind='avg')
    return (superpixel_img * 255).astype(np.uint8), segments

def extract_tumor_region(img, threshold=200):
    """Detect tumor and draw contour only for unhealthy regions."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()
    if len(contours) > 0:
        cv2.drawContours(img_copy, contours, -1, (255,0,0), 2)
    return img_copy

def build_superpixel_graph(img, segments):
    """
    Convert superpixel image to graph for GNN.
    Each superpixel is a node with RGB mean as feature.
    Edges connect adjacent superpixels.
    """
    import networkx as nx
    h, w = segments.shape
    G = nx.Graph()
    superpixels = np.unique(segments)
    for sp in superpixels:
        mask = (segments == sp)
        rgb_mean = img[mask].mean(axis=0)
        G.add_node(sp, feature=rgb_mean)

    # Add edges between neighboring superpixels
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    for y in range(h):
        for x in range(w):
            current = segments[y,x]
            for k in range(4):
                nx_, ny_ = x + dx[k], y + dy[k]
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    neighbor = segments[ny_, nx_]
                    if neighbor != current:
                        G.add_edge(current, neighbor)
    return G