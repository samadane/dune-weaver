"""
Dune Weaver - Image/SVG to .thr Pattern Converter
=================================================
Converts a raster drawing or SVG line art into a .thr polar coordinate file.
Raster input is skeletonized and traced. SVG input is sampled directly from
vector paths. A preview image can be generated to show traced paths and bridge
connections before using the resulting pattern.
"""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from svgpathtools import svg2paths2


def load_raster_mask(image_path: str, threshold: int, use_edges: bool) -> tuple[np.ndarray, Image.Image]:
    """Return a boolean 2D mask and grayscale source image for raster input."""
    source = Image.open(image_path).convert("L")

    if use_edges:
        filtered = source.filter(ImageFilter.FIND_EDGES)
        filtered = ImageOps.autocontrast(filtered)
        array = np.array(filtered)
        mask = array > threshold
    else:
        array = np.array(source)
        mask = array < threshold

    if not np.any(mask):
        print(f"No line pixels found. Try adjusting --threshold (currently {threshold}).")
        sys.exit(1)

    return mask.astype(bool), source


def _neighbors_8(mask: np.ndarray, y: int, x: int) -> list[int]:
    return [
        int(mask[y - 1, x]),
        int(mask[y - 1, x + 1]),
        int(mask[y, x + 1]),
        int(mask[y + 1, x + 1]),
        int(mask[y + 1, x]),
        int(mask[y + 1, x - 1]),
        int(mask[y, x - 1]),
        int(mask[y - 1, x - 1]),
    ]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1-pixel skeleton using Zhang-Suen thinning."""
    work = mask.copy().astype(np.uint8)
    if work.shape[0] < 3 or work.shape[1] < 3:
        return work.astype(bool)

    changed = True
    while changed:
        changed = False
        to_remove = []

        for y in range(1, work.shape[0] - 1):
            for x in range(1, work.shape[1] - 1):
                if work[y, x] != 1:
                    continue

                neighbors = _neighbors_8(work, y, x)
                count = sum(neighbors)
                transitions = sum(
                    1
                    for index in range(8)
                    if neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
                )

                if not (2 <= count <= 6 and transitions == 1):
                    continue
                if neighbors[0] * neighbors[2] * neighbors[4] != 0:
                    continue
                if neighbors[2] * neighbors[4] * neighbors[6] != 0:
                    continue
                to_remove.append((y, x))

        if to_remove:
            changed = True
            for y, x in to_remove:
                work[y, x] = 0

        to_remove = []
        for y in range(1, work.shape[0] - 1):
            for x in range(1, work.shape[1] - 1):
                if work[y, x] != 1:
                    continue

                neighbors = _neighbors_8(work, y, x)
                count = sum(neighbors)
                transitions = sum(
                    1
                    for index in range(8)
                    if neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
                )

                if not (2 <= count <= 6 and transitions == 1):
                    continue
                if neighbors[0] * neighbors[2] * neighbors[6] != 0:
                    continue
                if neighbors[0] * neighbors[4] * neighbors[6] != 0:
                    continue
                to_remove.append((y, x))

        if to_remove:
            changed = True
            for y, x in to_remove:
                work[y, x] = 0

    return work.astype(bool)


def _neighbor_coords(mask: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx = x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors


def _trace_segment(
    skeleton: np.ndarray,
    start: tuple[int, int],
    next_node: tuple[int, int],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
    node_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    path = [start]
    previous = start
    current = next_node

    while True:
        edge = tuple(sorted((previous, current)))
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        path.append(current)

        if current in node_set and current != start:
            break

        candidates = [point for point in _neighbor_coords(skeleton, current[0], current[1]) if point != previous]
        if not candidates:
            break

        unvisited = [
            point for point in candidates if tuple(sorted((current, point))) not in visited_edges
        ]
        next_point = unvisited[0] if unvisited else candidates[0]
        previous, current = current, next_point

    return path


def trace_skeleton_segments(skeleton: np.ndarray) -> list[np.ndarray]:
    """Trace skeleton branches into ordered x/y polylines."""
    coords = [tuple(coord) for coord in np.argwhere(skeleton)]
    if not coords:
        return []

    degree = {coord: len(_neighbor_coords(skeleton, coord[0], coord[1])) for coord in coords}
    node_set = {coord for coord, value in degree.items() if value != 2}
    if not node_set:
        node_set = {coords[0]}

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[np.ndarray] = []

    for node in sorted(node_set):
        for neighbor in _neighbor_coords(skeleton, node[0], node[1]):
            edge = tuple(sorted((node, neighbor)))
            if edge in visited_edges:
                continue
            segment = _trace_segment(skeleton, node, neighbor, visited_edges, node_set)
            if len(segment) > 1:
                segments.append(np.array([[point[1], point[0]] for point in segment], dtype=float))

    covered = {point for segment in segments for point in ((int(p[1]), int(p[0])) for p in segment)}
    leftovers = [coord for coord in coords if coord not in covered]
    for coord in leftovers:
        loop = [coord]
        covered.add(coord)
        neighbors = _neighbor_coords(skeleton, coord[0], coord[1])
        if not neighbors:
            segments.append(np.array([[coord[1], coord[0]]], dtype=float))
            continue

        previous = coord
        current = neighbors[0]
        while current not in covered:
            loop.append(current)
            covered.add(current)
            candidates = [point for point in _neighbor_coords(skeleton, current[0], current[1]) if point != previous]
            if not candidates:
                break
            previous, current = current, candidates[0]

        if len(loop) > 1:
            segments.append(np.array([[point[1], point[0]] for point in loop], dtype=float))

    return [segment for segment in segments if len(segment) > 0]


def _parse_svg_length(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", raw_value)
    return float(match.group(0)) if match else None


def _svg_dimensions(svg_path: str) -> tuple[float, float, float, float]:
    root = ET.parse(svg_path).getroot()
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [float(part) for part in view_box.replace(",", " ").split()]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return values[0], values[1], values[2], values[3]

    width = _parse_svg_length(root.attrib.get("width"))
    height = _parse_svg_length(root.attrib.get("height"))
    if width and height and width > 0 and height > 0:
        return 0.0, 0.0, width, height

    return 0.0, 0.0, 1000.0, 1000.0


def load_svg_segments(svg_path: str, samples: int) -> tuple[list[np.ndarray], tuple[int, int], Image.Image]:
    """Sample SVG paths into ordered x/y segments."""
    min_x, min_y, width, height = _svg_dimensions(svg_path)
    paths, _, _ = svg2paths2(svg_path)
    if not paths:
        print("No SVG paths found.")
        sys.exit(1)

    lengths = [max(path.length(error=1e-4), 1.0) for path in paths]
    total_length = sum(lengths)
    total_budget = max(samples, len(paths) * 8)

    segments: list[np.ndarray] = []
    for path, length in zip(paths, lengths):
        segment_points = max(8, int(round(total_budget * (length / total_length))))
        t_values = np.linspace(0.0, 1.0, segment_points)
        sampled = []
        for value in t_values:
            point = path.point(float(value))
            sampled.append([point.real - min_x, point.imag - min_y])
        segment = np.array(sampled, dtype=float)
        if len(segment) > 1:
            deduped = [segment[0]]
            for point in segment[1:]:
                if np.linalg.norm(point - deduped[-1]) > 1e-6:
                    deduped.append(point)
            segments.append(np.array(deduped, dtype=float))

    preview = Image.new("RGB", (max(1, int(math.ceil(width))), max(1, int(math.ceil(height)))), "white")
    return segments, preview.size, preview


def _bridge_points(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    distance = float(np.linalg.norm(end - start))
    steps = max(2, int(math.ceil(distance)))
    t_values = np.linspace(0.0, 1.0, steps)
    return np.array([start + (end - start) * value for value in t_values], dtype=float)


def connect_segments(
    segments: list[np.ndarray],
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Connect traced segments into a single path.

    Starts near the canvas center, then repeatedly chooses the next segment
    whose endpoint is nearest to the current tail. Disconnected shapes are
    joined with explicit straight bridge segments so the travel path is clear.
    """
    if not segments:
        return np.empty((0, 2), dtype=float), [], []

    center = np.array([canvas_size[0] / 2.0, canvas_size[1] / 2.0], dtype=float)
    remaining = [segment.copy() for segment in segments]

    start_index = 0
    start_flip = False
    best_start_distance = None
    for index, segment in enumerate(remaining):
        first_distance = np.linalg.norm(segment[0] - center)
        last_distance = np.linalg.norm(segment[-1] - center)
        if best_start_distance is None or first_distance < best_start_distance:
            best_start_distance = first_distance
            start_index = index
            start_flip = False
        if last_distance < best_start_distance:
            best_start_distance = last_distance
            start_index = index
            start_flip = True

    current = remaining.pop(start_index)
    if start_flip:
        current = current[::-1]

    ordered_segments = [current]
    bridges: list[np.ndarray] = []
    path_parts = [current]

    while remaining:
        tail = ordered_segments[-1][-1]
        best_index = 0
        best_flip = False
        best_score = None

        for index, segment in enumerate(remaining):
            start = segment[0]
            end = segment[-1]
            start_distance = np.linalg.norm(start - tail)
            end_distance = np.linalg.norm(end - tail)

            start_score = start_distance + 0.05 * np.linalg.norm(start - center)
            end_score = end_distance + 0.05 * np.linalg.norm(end - center)

            if best_score is None or start_score < best_score:
                best_score = start_score
                best_index = index
                best_flip = False
            if end_score < best_score:
                best_score = end_score
                best_index = index
                best_flip = True

        next_segment = remaining.pop(best_index)
        if best_flip:
            next_segment = next_segment[::-1]

        gap = np.linalg.norm(next_segment[0] - tail)
        if gap > 1.5:
            bridge = _bridge_points(tail, next_segment[0])
            bridges.append(bridge)
            path_parts.append(bridge[1:])

        ordered_segments.append(next_segment)
        path_parts.append(next_segment[1:])

    return np.vstack(path_parts), ordered_segments, bridges


def reduce_path(path: np.ndarray, target: int) -> np.ndarray:
    if len(path) <= target:
        return path
    keep = np.linspace(0, len(path) - 1, target, dtype=int)
    return path[keep]


def to_polar(path: np.ndarray, img_width: int, img_height: int):
    """
    Convert x/y coordinates to polar pairs.

    - Image center -> rho = 0
    - Furthest point from center -> rho = 1
    - theta is continuous (unwrapped) so the ball does not spin wildly
    """
    cx = img_width / 2.0
    cy = img_height / 2.0
    max_radius = math.sqrt(cx**2 + cy**2)

    xs = path[:, 0].astype(float) - cx
    ys = -(path[:, 1].astype(float) - cy)

    raw_theta = np.arctan2(ys, xs)
    rho = np.sqrt(xs**2 + ys**2) / max_radius
    rho = np.clip(rho, 0.0, 1.0)

    theta = np.unwrap(raw_theta)
    theta = theta - theta[0]
    return theta, rho


def write_thr(theta: np.ndarray, rho: np.ndarray, output_path: str):
    with open(output_path, "w") as handle:
        for theta_value, rho_value in zip(theta, rho):
            handle.write(f"{theta_value:.4f} {rho_value:.4f}\n")


def save_preview(
    base_image: Image.Image,
    ordered_segments: list[np.ndarray],
    bridges: list[np.ndarray],
    preview_path: str,
):
    preview = base_image.convert("RGB")
    draw = ImageDraw.Draw(preview)

    for segment in ordered_segments:
        if len(segment) >= 2:
            draw.line([tuple(point) for point in segment], fill=(0, 102, 204), width=2)

    for bridge in bridges:
        if len(bridge) >= 2:
            draw.line([tuple(point) for point in bridge], fill=(220, 64, 64), width=1)

    if ordered_segments:
        start = ordered_segments[0][0]
        end = ordered_segments[-1][-1]
        draw.ellipse((start[0] - 4, start[1] - 4, start[0] + 4, start[1] + 4), fill=(40, 180, 80))
        draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=(0, 0, 0))

    preview.save(preview_path)


def load_segments(
    input_path: str,
    threshold: int,
    use_edges: bool,
    samples: int,
) -> tuple[list[np.ndarray], tuple[int, int], Image.Image]:
    extension = os.path.splitext(input_path)[1].lower()
    if extension == ".svg":
        return load_svg_segments(input_path, samples)

    mask, source = load_raster_mask(input_path, threshold, use_edges)
    skeleton = skeletonize(mask)
    segments = trace_skeleton_segments(skeleton)
    return segments, source.size, source.convert("RGB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a raster image or SVG line art into a Dune Weaver .thr pattern file."
    )
    parser.add_argument("input", help="Path to input image or SVG file")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output filename without .thr extension (default: input filename)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=128,
        help="Pixel brightness threshold 0-255. Pixels darker than this are treated as lines for raster inputs. (default: 128)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=2000,
        help="Max number of traced path points to keep. More = finer detail but slower to draw. (default: 2000)",
    )
    parser.add_argument(
        "--edges",
        action="store_true",
        help="Use edge detection instead of threshold for raster inputs.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a preview PNG showing traced paths in blue and bridge connections in red.",
    )
    parser.add_argument(
        "--preview-path",
        default=None,
        help="Optional explicit path for the preview PNG. Defaults to patterns/<output>_preview.png.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(args.input))[0]
    out_name = (args.output or base).removesuffix(".thr")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "patterns")
    output_path = os.path.join(output_dir, out_name + ".thr")
    preview_path = args.preview_path or os.path.join(output_dir, out_name + "_preview.png")

    print("\n=== Image/SVG -> THR Converter ===")
    print(f"  Input    : {args.input}")
    print(f"  Output   : patterns/{out_name}.thr")
    print(f"  Preview  : {'enabled' if args.preview else 'disabled'}")
    if os.path.splitext(args.input)[1].lower() != ".svg":
        print(f"  Mode     : {'Edge detection' if args.edges else f'Threshold < {args.threshold}'}")
    print(f"  Samples  : up to {args.samples} points\n")

    segments, canvas_size, preview_image = load_segments(
        args.input,
        args.threshold,
        args.edges,
        args.samples,
    )
    print(f"  Canvas size: {canvas_size[0]} x {canvas_size[1]} px")
    print(f"  Loaded {len(segments)} traced segments.")

    path, ordered_segments, bridges = connect_segments(segments, canvas_size)
    path = reduce_path(path, args.samples)
    print(f"  Joined into {len(ordered_segments)} ordered segments with {len(bridges)} bridge connections.")
    print(f"  Final traced path has {len(path)} points.")

    theta, rho = to_polar(path, canvas_size[0], canvas_size[1])
    write_thr(theta, rho, output_path)
    print(f"\nSaved {len(theta)} points to patterns/{out_name}.thr")

    if args.preview:
        save_preview(preview_image, ordered_segments, bridges, preview_path)
        print(f"Saved preview to {preview_path}")

    print("Refresh the Dune Weaver UI to see your pattern.\n")


if __name__ == "__main__":
    main()
