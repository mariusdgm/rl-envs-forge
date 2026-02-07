import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

sns.set_theme()


def centrality_to_sizes_area(
    centralities,
    *,
    min_size=300,
    max_size=2000,
    eps=1e-12,
    clip_quantile=0.98,
    gamma=1.0,
):
    c = np.asarray(centralities, dtype=float)
    sizes = np.zeros_like(c, dtype=float)

    mask = c > eps
    if not np.any(mask):
        return sizes

    vals = c[mask].copy()
    hi = np.quantile(vals, clip_quantile)
    vals = np.minimum(vals, hi)

    c_min = vals.min()
    c_max = vals.max()
    if np.isclose(c_min, c_max):
        sizes[mask] = min_size
        return sizes

    z = (vals - c_min) / (c_max - c_min)
    z = np.power(z, gamma)
    sizes[mask] = min_size + z * (max_size - min_size)
    return sizes


def _spread_positions(pos, node_radii, max_iter=300, pad=0.02, step=0.35):
    nodes = list(pos.keys())
    P = np.array([pos[n] for n in nodes], dtype=float)

    for _ in range(max_iter):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                d = P[j] - P[i]
                dist = np.linalg.norm(d) + 1e-12
                min_dist = node_radii[nodes[i]] + node_radii[nodes[j]] + pad
                if dist < min_dist:
                    push = (min_dist - dist) * (d / dist)
                    P[i] -= 0.5 * step * push
                    P[j] += 0.5 * step * push
                    moved = True
        if not moved:
            break

    return {n: P[k] for k, n in enumerate(nodes)}


def _radius_points_from_area(area_pt2: float) -> float:
    return 0.0 if area_pt2 <= 0 else float(np.sqrt(area_pt2 / np.pi))


def _fit_positions_to_axes_with_node_radii(
    pos,
    *,
    node_size_map,
    ax,
    pad_frac=0.03,
    extra_pts=2.0,
    spread_factor=1.0,  # can be >1, but will be safely re-normalized
    safety_shrink=0.98,  # final tiny shrink to be extra safe
):
    """
    Center + spread + fit so that circles (not just centers) stay inside [-1,1]^2.

    IMPORTANT FIX vs earlier version:
      If spread_factor > 1, we re-normalize after spreading, so positions can never
      overflow purely due to spread.
    """
    fig = ax.figure
    fig.canvas.draw()  # bbox is valid (includes title/subplots_adjust)

    nodes = list(pos.keys())
    P = np.array([pos[n] for n in nodes], dtype=float)

    # center
    P -= P.mean(axis=0, keepdims=True)

    # normalize to max_abs=1
    max_abs = np.max(np.abs(P)) + 1e-12
    P = P / max_abs

    # spread (then re-normalize so max_abs=1 again)
    P *= float(spread_factor)
    max_abs2 = np.max(np.abs(P)) + 1e-12
    P = P / max_abs2

    # axes size in pixels
    bbox = ax.get_window_extent()
    ax_w_px = max(bbox.width, 1.0)
    ax_h_px = max(bbox.height, 1.0)

    dpi = fig.dpi
    pt_to_px = dpi / 72.0

    # if we set limits to [-1,1], then pixels per data unit:
    px_per_data_x = ax_w_px / 2.0
    px_per_data_y = ax_h_px / 2.0

    # compute per-node radius in data units
    r_data_x = []
    r_data_y = []
    for n in nodes:
        area = float(node_size_map.get(n, 0.0))
        r_pt = _radius_points_from_area(area) + float(extra_pts)
        r_px = r_pt * pt_to_px
        r_data_x.append(r_px / px_per_data_x)
        r_data_y.append(r_px / px_per_data_y)

    r_data_x = np.array(r_data_x) if len(r_data_x) else np.array([0.0])
    r_data_y = np.array(r_data_y) if len(r_data_y) else np.array([0.0])

    margin_x = float(np.max(r_data_x)) + float(pad_frac)
    margin_y = float(np.max(r_data_y)) + float(pad_frac)

    # how much room we have for centers
    room = max(min(1.0 - margin_x, 1.0 - margin_y), 0.05)

    P *= room * float(safety_shrink)

    return {n: P[i] for i, n in enumerate(nodes)}


def _isotropize_positions(pos, min_std=1e-6):
    """
    Make the layout fill the plane more by equalizing x/y spread.
    This fixes the 'long line / oval' effect from spring_layout.
    """
    nodes = list(pos.keys())
    P = np.array([pos[n] for n in nodes], dtype=float)

    # center
    P -= P.mean(axis=0, keepdims=True)

    # scale each axis to unit std (avoid division by ~0)
    std = P.std(axis=0)
    std = np.where(std < min_std, 1.0, std)
    P = P / std

    return {n: P[i] for i, n in enumerate(nodes)}


def _layout_is_too_collinear(pos, ratio_thresh=0.12):
    """
    Detect skinny layouts using PCA eigenvalue ratio.
    If smallest/ largest variance < ratio_thresh => basically a line.
    """
    nodes = list(pos.keys())
    P = np.array([pos[n] for n in nodes], dtype=float)
    P -= P.mean(axis=0, keepdims=True)

    # covariance eigenvalues
    C = (P.T @ P) / max(len(P), 1)
    w = np.linalg.eigvalsh(C)  # sorted
    if w[-1] <= 1e-12:
        return True
    return (w[0] / w[-1]) < ratio_thresh


def _jitter_positions(pos, rng, scale=0.15):
    nodes = list(pos.keys())
    P = np.array([pos[n] for n in nodes], dtype=float)
    P = P + rng.normal(scale=scale, size=P.shape)
    return {n: P[i] for i, n in enumerate(nodes)}


def draw_network_graph(
    adjacency_matrix,
    node_centralities,
    *,
    # node sizing (area-based)
    min_size=300,
    max_size=2000,
    gamma=1.0,
    clip_quantile=0.98,
    eps=1e-12,
    # layout/spacing
    seed=0,  # base seed (e.g., env seed)
    layout_try=0,
    randomize_layout=False,
    iterations=1200,
    spread=True,
    spread_factor=1.5,
    # overlap resolver
    spread_pad=0.03,
    # visuals
    node_color="skyblue",
    edge_color="gray",
    node_alpha=0.85,
    edge_alpha=0.85,
    font_size=12,
    arrowsize=18,
    edge_lw=1.1,
    edge_pad_pt=2.0,
    curve_bidirectional=True,
    curve_rad=0.18,
    # title="Network Graph",
):
    """
    Draw a directed network graph with:
      - node area ~ centrality (bounded), zeros invisible (size 0)
      - multiple layout retries for the *same* graph by varying `layout_try`
      - nodes kept fully inside plot bounds (circle radius included)
      - arrowheads stop at node boundaries (FancyArrowPatch shrinkA/B)
      - optional separation pass to reduce overlaps

    Usage:
      draw_network_graph(A, c, seed=env.seed, layout_try=0)
      draw_network_graph(A, c, seed=env.seed, layout_try=1)
      draw_network_graph(A, c, randomize_layout=True)
    """
    G = nx.DiGraph(adjacency_matrix)
    nodes = list(G.nodes())

    # --- Node sizes (points^2) ---
    sizes = centrality_to_sizes_area(
        node_centralities,
        min_size=min_size,
        max_size=max_size,
        gamma=gamma,
        clip_quantile=clip_quantile,
        eps=eps,
    )
    size_map = {node: float(sizes[i]) for i, node in enumerate(nodes)}

    # --- Figure/Axes FIRST (so fitting uses final axes geometry) ---
    fig, ax = plt.subplots(figsize=(6, 5), dpi=2000)
    ax.set_axis_off()
    # ax.set_title(title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # --- Layout: stronger repulsion for fewer crowded arrows ---
    n = max(len(G), 2)
    r_max = np.sqrt(float(np.max(sizes))) if np.max(sizes) > 0 else 0.0
    k_base = 3.0 / np.sqrt(n)
    inflate = 1.0 + min(3.0, r_max / 24.0)
    k = k_base * inflate

    # --- Layout seed control (ONLY affects plotting) ---
    if randomize_layout:
        layout_seed = int(
            np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint32)
        )
    else:
        layout_seed = int(seed) + 10_000 * int(layout_try + 1)

    layout_seed = int(layout_seed % (2**32))  # paranoia: keep in 32-bit range
    rng = np.random.default_rng(layout_seed)

    # Randomize k a bit to escape same local minima
    k_jitter = float(rng.uniform(0.80, 1.25)) if randomize_layout else 1.0
    k_used = k * k_jitter

    # Random initial positions (THIS is the big difference)
    init_pos_arr = rng.normal(size=(len(nodes), 2))
    init_pos = {node: init_pos_arr[i] for i, node in enumerate(nodes)}

    pos = nx.spring_layout(
        G,
        k=k_used,
        iterations=int(iterations),
        seed=layout_seed,
        pos=init_pos,  # <--- key
    )

    pos = _isotropize_positions(pos)

    if _layout_is_too_collinear(pos, ratio_thresh=0.12):
        pos = _jitter_positions(pos, rng, scale=0.20)

        # re-run spring from jittered pos (short, just to settle)
        pos = nx.spring_layout(
            G,
            k=k_used,
            iterations=max(200, int(iterations * 0.35)),
            seed=layout_seed,
            pos=pos,
        )

        pos = _isotropize_positions(pos)

    if _layout_is_too_collinear(pos, ratio_thresh=0.08):
        pos = nx.kamada_kawai_layout(G)
        pos = _isotropize_positions(pos)

    # Fit once (keeps circles in bounds, not just centers)
    pos = _fit_positions_to_axes_with_node_radii(
        pos,
        node_size_map=size_map,
        ax=ax,
        pad_frac=0.03,
        extra_pts=2.0,
        spread_factor=spread_factor,
        safety_shrink=0.985,
    )

    # Optional overlap resolver then refit
    if spread and np.max(sizes) > 0:
        max_s = float(np.max(sizes))
        layout_r = 0.03 + 0.10 * (np.sqrt(sizes) / (np.sqrt(max_s) + 1e-12))
        node_radii = {node: float(layout_r[i]) for i, node in enumerate(nodes)}

        pos = _spread_positions(
            pos,
            node_radii,
            max_iter=350,
            pad=spread_pad,
            step=0.40,
        )

        pos = _fit_positions_to_axes_with_node_radii(
            pos,
            node_size_map=size_map,
            ax=ax,
            pad_frac=0.03,
            extra_pts=2.0,
            spread_factor=spread_factor,
            safety_shrink=0.985,
        )

    # Lock bounds BEFORE drawing so everything is consistent with fitting
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    # --- Draw nodes (zeros => no circle) ---
    nonzero_nodes = [node for node in nodes if size_map[node] > 0.0]
    nonzero_sizes = [size_map[node] for node in nonzero_nodes]

    if nonzero_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nonzero_nodes,
            node_size=nonzero_sizes,
            node_color=node_color,
            alpha=node_alpha,
            edgecolors="black",
            linewidths=1.0,
            ax=ax,
        )

    # Labels for all nodes
    nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight="bold", ax=ax)

    # --- Draw edges clipped to node boundaries ---
    rad_pt = {node: _radius_points_from_area(size_map[node]) for node in nodes}
    edge_set = set(G.edges())

    for u, v in G.edges():
        if u == v:
            continue

        # curve reciprocal edges to reduce overlap
        rad = 0.0
        if curve_bidirectional and (v, u) in edge_set and u < v:
            rad = curve_rad
        elif curve_bidirectional and (v, u) in edge_set and u > v:
            rad = -curve_rad

        arrow = FancyArrowPatch(
            posA=pos[u],
            posB=pos[v],
            arrowstyle="-|>",
            mutation_scale=arrowsize,
            color=edge_color,
            linewidth=edge_lw,
            alpha=edge_alpha,
            shrinkA=rad_pt.get(u, 0.0) + edge_pad_pt,
            shrinkB=rad_pt.get(v, 0.0) + edge_pad_pt,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    # fig.tight_layout(pad=0)
    # ax.set_axis_off()
    # ax.set_position([0, 0, 1, 1])

    plt.show()
    return fig


def plot_centralities_sorted(eigv):
    indices = np.arange(len(eigv))
    sorted_indices = np.argsort(eigv)[::-1]
    sorted_centralities = eigv[sorted_indices]
    sorted_labels = [str(i) for i in indices[sorted_indices]]

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_labels, y=sorted_centralities, palette="Blues_d")
    plt.xlabel("Node Index")
    plt.ylabel("Centrality")
    plt.title("Centralities Sorted in Decreasing Order")
    plt.xticks(rotation=45)
    plt.show()
    
    return fig
