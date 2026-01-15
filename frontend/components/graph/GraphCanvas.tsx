"use client";

import { useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Panel,
  type Node,
  type Edge,
  type NodeMouseHandler,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { TerminalNode } from "./TerminalNode";
import { getHighlightedEdgeStyle, getDefaultEdgeStyle } from "@/lib/graphUtils";

const nodeTypes = {
  terminal: TerminalNode,
};

interface GraphCanvasProps {
  initialNodes: Node[];
  initialEdges: Edge[];
  onNodeClick: (nodeId: string) => void;
  selectedNodeIds: string[];
  highlightedEdgeIds: string[];
}

export const GraphCanvas = ({
  initialNodes,
  initialEdges,
  onNodeClick,
  selectedNodeIds,
  highlightedEdgeIds,
}: GraphCanvasProps) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when initial data changes
  useEffect(() => {
    setNodes(initialNodes);
  }, [initialNodes, setNodes]);

  // Update edges when initial data changes
  useEffect(() => {
    setEdges(initialEdges);
  }, [initialEdges, setEdges]);

  // Update node selection states
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        selected: selectedNodeIds.includes(node.id),
      }))
    );
  }, [selectedNodeIds, setNodes]);

  // Update edge highlighting
  useEffect(() => {
    setEdges((eds) =>
      eds.map((edge) => ({
        ...edge,
        style: highlightedEdgeIds.includes(edge.id)
          ? getHighlightedEdgeStyle()
          : getDefaultEdgeStyle(),
        animated: highlightedEdgeIds.includes(edge.id),
      }))
    );
  }, [highlightedEdgeIds, setEdges]);

  const handleNodeClick: NodeMouseHandler = useCallback(
    (_, node) => {
      onNodeClick(node.id);
    },
    [onNodeClick]
  );

  const defaultEdgeOptions = useMemo(
    () => ({
      type: "smoothstep",
      animated: false,
      style: getDefaultEdgeStyle(),
      markerEnd: {
        type: "arrowclosed" as const,
        color: "#1f521f",
        width: 20,
        height: 20,
      },
    }),
    []
  );

  return (
    <div className="w-full h-full bg-terminal-black">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        attributionPosition="bottom-left"
        className="bg-terminal-black"
        proOptions={{ hideAttribution: true }}
      >
        {/* Terminal-styled controls */}
        <Controls
          className="!bg-terminal-black !border !border-terminal-green-muted"
          style={{
            button: {
              backgroundColor: "#0a0a0a",
              borderColor: "#1f521f",
              color: "#33ff00",
            },
          }}
        />

        {/* Grid background (terminal style) */}
        <Background
          color="#1f521f"
          gap={16}
          variant={BackgroundVariant.Dots}
          size={1}
        />

        {/* Legend Panel */}
        <Panel position="top-right" className="terminal-window text-xs">
          <div className="space-y-1">
            <div className="text-terminal-green opacity-70">LEGEND:</div>
            <div className="text-terminal-green">[→] DIRECTED</div>
            <div className="text-terminal-green">[↔] BIDIRECTED</div>
            <div className="text-terminal-green">[─] UNDIRECTED</div>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
};
