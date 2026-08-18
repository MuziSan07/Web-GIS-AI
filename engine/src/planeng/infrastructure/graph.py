"""The service corridor graph.

Water mains, sewers, electrical reticulation and street lighting all run inside
the road reserve. That is not a simplification for the model's convenience - it
is how services are actually laid, because the reserve is the only land the
authority owns in perpetuity and the only place a utility can dig without
negotiating a servitude across a private plot.

So the four networks are one routing problem over one graph, solved four times
with different sources and different costs. Building that graph once here keeps
the networks consistent with each other: a main and a sewer in the same street
are the same edge, which is what lets the corridor cross-section be checked.

Nodes are the road centreline intersections and vertices; edges are the runs
between them. `unary_union` does the noding, so two roads that cross without a
shared vertex still produce a junction node rather than two networks that pass
through each other and never connect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from ..site.model import Site

# Coordinates within this distance are the same junction. Centrelines are
# traced through raster cell centres, so endpoints that should coincide can be
# a few centimetres apart; without snapping the graph silently disconnects.
SNAP_M = 0.5


@dataclass
class ServiceGraph:
    """An undirected graph of the road reserve, ready to route services over."""

    nodes: np.ndarray                       # (n, 2) easting/northing
    edges: list[tuple[int, int]] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    road_class: list[str] = field(default_factory=list)
    elevation: np.ndarray | None = None     # metres at each node
    _lookup: dict | None = field(default=None, repr=False, compare=False)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def point(self, i: int) -> Point:
        return Point(self.nodes[i][0], self.nodes[i][1])

    def edge_line(self, e: int) -> LineString:
        a, b = self.edges[e]
        return LineString([tuple(self.nodes[a]), tuple(self.nodes[b])])

    def nearest_node(self, geom) -> int:
        """The graph node closest to a point or polygon centroid.

        Every service has to reach land, and land is not on the graph. This is
        the connection point - the erf connection, the reservoir take-off, the
        transformer position.
        """
        pt = geom if geom.geom_type == "Point" else geom.centroid
        d = np.hypot(self.nodes[:, 0] - pt.x, self.nodes[:, 1] - pt.y)
        return int(np.argmin(d))

    def matrix(self, weights=None, directed: bool = False) -> csr_matrix:
        w = self.lengths if weights is None else weights
        rows, cols, vals = [], [], []
        for (a, b), cost in zip(self.edges, w):
            rows.append(a)
            cols.append(b)
            vals.append(float(cost))
            if not directed:
                rows.append(b)
                cols.append(a)
                vals.append(float(cost))
        if not rows:
            return csr_matrix((self.n_nodes, self.n_nodes))
        return csr_matrix((vals, (rows, cols)),
                          shape=(self.n_nodes, self.n_nodes))

    def tree_from(self, source: int, weights=None,
                  directed: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Shortest-path tree. Returns (distance, predecessor) per node.

        A shortest-path tree, not a minimum spanning tree: for a distribution
        network the quantity that matters is the run from the source to each
        consumer - pressure loss, voltage drop, sewer fall - not the total
        length of pipe. An MST minimises the wrong thing and can hang a whole
        neighbourhood off the end of a long chain.
        """
        dist, pred = dijkstra(self.matrix(weights, directed=directed),
                              directed=directed, indices=source,
                              return_predecessors=True)
        return dist, pred

    def path_edges(self, pred: np.ndarray, target: int) -> list[tuple[int, int]]:
        """Walk a predecessor array back to the source."""
        out = []
        cur = int(target)
        guard = 0
        while pred[cur] >= 0 and guard < self.n_nodes + 1:
            out.append((int(pred[cur]), cur))
            cur = int(pred[cur])
            guard += 1
        return out[::-1]

    def class_of(self, a: int, b: int) -> str:
        """Road class of the edge between two nodes.

        Cached: lighting and corridor checks ask this once per run, and a linear
        scan of every edge each time turns a cheap lookup into the slowest part
        of the layer.
        """
        if getattr(self, "_lookup", None) is None:
            self._lookup = {
                (min(u, v), max(u, v)): i
                for i, (u, v) in enumerate(self.edges)
            }
        i = self._lookup.get((min(a, b), max(a, b)))
        return self.road_class[i] if i is not None else "local_access"

    def summary(self) -> dict:
        return {
            "nodes": self.n_nodes,
            "edges": len(self.edges),
            "total_length_km": round(sum(self.lengths) / 1000, 2),
        }


def build_graph(network, site: Site) -> ServiceGraph:
    """Turn the road network into a routable graph.

    Nodes carry ground elevation sampled from the DEM. Sewers are the reason: a
    graph without levels can route a gravity main uphill and never notice.
    """
    lines, classes = [], []
    for seg in network.segments:
        geom = seg.geometry
        if geom.is_empty or geom.length <= 0:
            continue
        lines.append(geom)
        classes.append(seg.road_class)

    if not lines:
        return ServiceGraph(nodes=np.zeros((0, 2)), elevation=np.zeros(0))

    # Node the network: roads that cross must share a junction, or services
    # routed over them pass through each other without ever connecting.
    noded = unary_union(lines)
    parts = list(noded.geoms) if isinstance(noded, MultiLineString) else [noded]

    # Which road class a run belongs to, for lighting and corridor width. The
    # union discards that attribute, so recover it by proximity to the
    # original. Indexed rather than scanned: this is asked once per noded run,
    # and comparing every run against every input segment made building the
    # graph the slowest step in the whole engineering layer.
    from shapely.strtree import STRtree
    tree = STRtree(lines)

    def classify(mid: Point) -> str:
        i = tree.nearest(mid)
        return classes[int(i)] if i is not None else "local_access"

    index: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []

    def node_id(xy) -> int:
        key = (int(round(xy[0] / SNAP_M)), int(round(xy[1] / SNAP_M)))
        if key not in index:
            index[key] = len(coords)
            coords.append((xy[0], xy[1]))
        return index[key]

    edges, lengths, edge_class = [], [], []
    seen: set[tuple[int, int]] = set()

    for part in parts:
        pts = list(part.coords)
        if len(pts) < 2:
            continue
        cls = classify(part.interpolate(0.5, normalized=True))
        for p, q in zip(pts, pts[1:]):
            a, b = node_id(p), node_id(q)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            edges.append((a, b))
            lengths.append(float(np.hypot(q[0] - p[0], q[1] - p[1])))
            edge_class.append(cls)

    nodes = np.asarray(coords, dtype=float)
    graph = ServiceGraph(
        nodes=nodes, edges=edges, lengths=lengths, road_class=edge_class,
        elevation=_sample_dem(site, nodes),
    )

    # Centrelines meet reserves, not each other. A local street that ends at the
    # kerb of a forty metre arterial stops twenty metres short of that
    # arterial's centreline, so the two are connected on the ground and
    # disconnected in this graph. Close those gaps before anything routes over
    # it, or the water network will decide half the town is unreachable.
    widest = max((s.reserve_width_m for s in network.segments), default=0.0)
    stitch(graph, max_gap=widest / 2 + 2.0)
    return graph


def stitch(graph: ServiceGraph, max_gap: float, max_rounds: int = 8) -> int:
    """Join graph components separated by less than a reserve half-width.

    A street rarely ends near another street's *vertex*. It ends part-way along
    another street's run - at the kerb, mid-block. So the connector is found by
    projecting the loose end onto the nearby edge and splitting that edge at the
    projection, which is also where the tee would physically be built.

    Only the single shortest connector between any two components is taken per
    round, and only across a gap no wider than half the widest road reserve.
    Joining every pair within tolerance would wire together streets that merely
    run near each other, inventing connectivity the layout does not have.

    Returns the number of connectors added.
    """
    from shapely.strtree import STRtree

    if graph.n_nodes < 2 or max_gap <= 0:
        return 0

    added = 0
    for _ in range(max_rounds):
        labels = components(graph)
        if len(labels) == 0 or int(labels.max()) + 1 <= 1:
            break

        lines = [graph.edge_line(e) for e in range(len(graph.edges))]
        edge_label = np.array([labels[a] for a, _ in graph.edges])
        tree = STRtree(lines)

        # Shortest candidate connector per pair of components.
        best: dict[tuple[int, int], tuple] = {}
        for i in range(graph.n_nodes):
            pt = Point(graph.nodes[i])
            for e in tree.query(pt.buffer(max_gap)):
                e = int(e)
                if edge_label[e] == labels[i]:
                    continue
                gap = lines[e].distance(pt)
                if gap > max_gap:
                    continue
                key = (min(labels[i], edge_label[e]),
                       max(labels[i], edge_label[e]))
                if key not in best or gap < best[key][0]:
                    best[key] = (float(gap), i, e)

        if not best:
            break

        # One split per edge per round: two splits of the same edge would use
        # stale endpoint indices.
        used_edges: set[int] = set()
        for gap, i, e in sorted(best.values()):
            if e in used_edges:
                continue
            used_edges.add(e)
            _connect_to_edge(graph, i, e, gap)
            added += 1

    return added


def _connect_to_edge(graph: ServiceGraph, node: int, edge: int,
                     gap: float) -> None:
    """Tee a loose node into an edge, splitting the edge at the tee."""
    line = graph.edge_line(edge)
    pt = Point(graph.nodes[node])
    along = line.project(pt)
    tee = line.interpolate(along)

    a, b = graph.edges[edge]

    # A tee that lands on an end of the run is that end. Splitting there anyway
    # produces a zero length edge, which is not a pipe, not a road and not a
    # gradient - and it reappears downstream as a run that fails every
    # dimensional check for reasons that have nothing to do with the design.
    for end in (a, b):
        if tee.distance(Point(graph.nodes[end])) <= SNAP_M:
            if end != node:
                graph.edges.append((node, end))
                graph.lengths.append(max(gap, 0.01))
                graph.road_class.append("connector")
            return
    total = max(graph.lengths[edge], 1e-9)
    cls = graph.road_class[edge]

    new = graph.n_nodes
    graph.nodes = np.vstack([graph.nodes, [[tee.x, tee.y]]])
    if graph.elevation is not None and len(graph.elevation):
        # Interpolate level along the run rather than resampling the DEM: the
        # tee sits on the road, whose level is already fixed by its endpoints.
        f = along / total
        graph.elevation = np.append(
            graph.elevation,
            graph.elevation[a] * (1 - f) + graph.elevation[b] * f)

    graph.edges[edge] = (a, new)
    graph.lengths[edge] = max(along, 0.01)
    graph.edges.append((new, b))
    graph.lengths.append(max(total - along, 0.01))
    graph.road_class.append(cls)

    graph.edges.append((node, new))
    graph.lengths.append(max(gap, 0.01))
    graph.road_class.append("connector")


def _sample_dem(site: Site, nodes: np.ndarray) -> np.ndarray:
    """Ground level at each node, from the site DEM."""
    if len(nodes) == 0:
        return np.zeros(0)
    inv = ~site.transform
    cols, rows = inv * (nodes[:, 0], nodes[:, 1])
    rows = np.clip(np.round(np.asarray(rows)).astype(int), 0, site.shape[0] - 1)
    cols = np.clip(np.round(np.asarray(cols)).astype(int), 0, site.shape[1] - 1)
    return site.dem[rows, cols].astype(float)


def components(graph: ServiceGraph) -> np.ndarray:
    """Connected-component label per node.

    A road network generated over difficult terrain is not guaranteed to be one
    piece. Routing services over a disconnected graph produces unreachable
    consumers, and it is better to say so than to draw a plausible-looking main
    that serves nobody.
    """
    if graph.n_nodes == 0:
        return np.zeros(0, dtype=int)
    _, labels = connected_components(graph.matrix(), directed=False)
    return labels
