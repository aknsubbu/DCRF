import type { Edge } from "@xyflow/react";

/**
 * Find all paths between selected nodes using BFS
 */
export function findAllPaths(
  edges: Edge[],
  selectedNodeIds: string[],
  options: { directOnly?: boolean; maxDepth?: number } = {}
): string[][] {
  const { directOnly = false, maxDepth = 5 } = options;
  const allPaths: string[][] = [];

  // Build adjacency list
  const graph = buildAdjacencyList(edges);

  // Find paths between each pair of selected nodes
  for (let i = 0; i < selectedNodeIds.length; i++) {
    for (let j = i + 1; j < selectedNodeIds.length; j++) {
      const source = selectedNodeIds[i];
      const target = selectedNodeIds[j];

      // Search in both directions
      const forwardPaths = bfsAllPaths(graph, source, target, maxDepth);
      const backwardPaths = bfsAllPaths(graph, target, source, maxDepth);

      allPaths.push(...forwardPaths);
      allPaths.push(...backwardPaths.map((p) => [...p].reverse()));
    }
  }

  // Filter for direct paths if requested
  if (directOnly) {
    return uniquePaths(allPaths.filter((path) => path.length === 2));
  }

  return uniquePaths(allPaths);
}

/**
 * Build adjacency list from edges
 */
function buildAdjacencyList(edges: Edge[]): Map<string, string[]> {
  const graph = new Map<string, string[]>();

  edges.forEach((edge) => {
    if (!graph.has(edge.source)) {
      graph.set(edge.source, []);
    }
    graph.get(edge.source)!.push(edge.target);
  });

  return graph;
}

/**
 * BFS to find all paths from start to end
 */
function bfsAllPaths(
  graph: Map<string, string[]>,
  start: string,
  end: string,
  maxDepth: number
): string[][] {
  const paths: string[][] = [];
  const queue: string[][] = [[start]];

  while (queue.length > 0) {
    const path = queue.shift()!;
    const node = path[path.length - 1];

    // Check depth limit
    if (path.length > maxDepth + 1) continue;

    // Found target
    if (node === end) {
      paths.push(path);
      continue;
    }

    // Explore neighbors
    const neighbors = graph.get(node) || [];
    for (const neighbor of neighbors) {
      // Avoid cycles
      if (!path.includes(neighbor)) {
        queue.push([...path, neighbor]);
      }
    }
  }

  return paths;
}

/**
 * Remove duplicate paths
 */
function uniquePaths(paths: string[][]): string[][] {
  const seen = new Set<string>();
  return paths.filter((path) => {
    const key = path.join("->");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/**
 * Extract edge IDs that are part of given paths
 */
export function getEdgeIdsFromPaths(
  paths: string[][],
  edges: Edge[]
): string[] {
  const edgeIds = new Set<string>();

  paths.forEach((path) => {
    for (let i = 0; i < path.length - 1; i++) {
      const source = path[i];
      const target = path[i + 1];

      // Find matching edge
      const edge = edges.find(
        (e) => e.source === source && e.target === target
      );

      if (edge) {
        edgeIds.add(edge.id);
      }
    }
  });

  return Array.from(edgeIds);
}

/**
 * Get node names from paths for display
 */
export function formatPathForDisplay(
  path: string[],
  nodeNames: string[]
): string {
  return path
    .map((nodeId) => {
      const index = parseInt(nodeId.split("-")[1], 10);
      return nodeNames[index] || nodeId;
    })
    .join(" → ");
}
