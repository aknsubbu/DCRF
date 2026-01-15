import dagre from "dagre";
import type { Node, Edge } from "@xyflow/react";

export interface MatrixData {
  nodeNames: string[];
  matrix: number[][];
}

/**
 * Convert adjacency matrix to React Flow nodes and edges
 */
export function adjacencyMatrixToGraph(
  nodeNames: string[],
  matrix: number[][]
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = nodeNames.map((name, index) => ({
    id: `node-${index}`,
    type: "terminal",
    data: { label: name },
    position: { x: 0, y: 0 }, // Will be set by layout
  }));

  const edges: Edge[] = [];

  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      if (matrix[i][j] !== 0 && i !== j) {
        const edgeType = getEdgeType(matrix[i][j], matrix[j]?.[i] || 0);

        edges.push({
          id: `edge-${i}-${j}`,
          source: `node-${i}`,
          target: `node-${j}`,
          type: "smoothstep",
          animated: false,
          style: {
            stroke: "#1f521f",
            strokeWidth: 2,
          },
          markerEnd: edgeType.hasArrow
            ? {
                type: "arrowclosed" as const,
                color: "#1f521f",
                width: 20,
                height: 20,
              }
            : undefined,
          data: { edgeType: edgeType.type },
        });
      }
    }
  }

  return { nodes, edges };
}

/**
 * Determine edge type based on matrix values
 * 1 = directed edge
 * 2 = bidirectional (both directions exist)
 */
function getEdgeType(
  forwardVal: number,
  backwardVal: number
): { type: string; hasArrow: boolean } {
  if (forwardVal > 0 && backwardVal > 0) {
    return { type: "bidirectional", hasArrow: true };
  }
  if (forwardVal > 0) {
    return { type: "directed", hasArrow: true };
  }
  return { type: "undirected", hasArrow: false };
}

/**
 * Apply Dagre layout to nodes
 */
export function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" | "BT" | "RL" = "TB"
): { nodes: Node[]; edges: Edge[] } {
  if (nodes.length === 0) {
    return { nodes: [], edges: [] };
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  const nodeWidth = 150;
  const nodeHeight = 60;

  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: 80,
    ranksep: 120,
  });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - nodeWidth / 2,
        y: nodeWithPosition.y - nodeHeight / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}

/**
 * Get highlighted edge style
 */
export function getHighlightedEdgeStyle() {
  return {
    stroke: "#33ff00",
    strokeWidth: 3,
    filter: "drop-shadow(0 0 4px rgba(51, 255, 0, 0.8))",
  };
}

/**
 * Get default edge style
 */
export function getDefaultEdgeStyle() {
  return {
    stroke: "#1f521f",
    strokeWidth: 2,
  };
}
