"use client";

import { useState, useCallback, useEffect } from "react";
import type { Node, Edge } from "@xyflow/react";

import {
  CRTOverlay,
  Header,
  StatusLog,
  getCurrentTime,
  UploadSection,
  MatrixPreview,
  GraphCanvas,
  SelectedNodesPanel,
  PathFinderPanel,
  type LogEntry,
} from "@/components/graph";

import { adjacencyMatrixToGraph, getLayoutedElements } from "@/lib/graphUtils";

import { findAllPaths, getEdgeIdsFromPaths } from "@/lib/pathFinding";

interface MatrixData {
  nodeNames: string[];
  matrix: number[][];
}

export default function Home() {
  // Data state
  const [nodeNames, setNodeNames] = useState<string[]>([]);
  const [matrix, setMatrix] = useState<number[][]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  // Interaction state
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [highlightedEdgeIds, setHighlightedEdgeIds] = useState<string[]>([]);

  // UI state
  const [uploadStatus, setUploadStatus] = useState({
    success: false,
    message: "",
  });
  const [paths, setPaths] = useState<string[][]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isMatrixExpanded, setIsMatrixExpanded] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([
    { timestamp: getCurrentTime(), type: "ok", message: "Application ready" },
  ]);

  // Check for mobile viewport
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedNodeIds([]);
        setHighlightedEdgeIds([]);
        addLog("info", "Selection cleared");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const addLog = useCallback((type: LogEntry["type"], message: string) => {
    setLogs((prev) => [
      ...prev,
      {
        timestamp: getCurrentTime(),
        type,
        message,
      },
    ]);
  }, []);

  const handleFileUpload = useCallback(
    (data: MatrixData) => {
      setNodeNames(data.nodeNames);
      setMatrix(data.matrix);

      addLog("ok", `Matrix loaded: ${data.nodeNames.length} nodes`);

      // Generate graph
      const { nodes: graphNodes, edges: graphEdges } = adjacencyMatrixToGraph(
        data.nodeNames,
        data.matrix
      );

      // Apply layout
      const { nodes: layoutedNodes, edges: layoutedEdges } =
        getLayoutedElements(graphNodes, graphEdges, "TB");

      setNodes(layoutedNodes);
      setEdges(layoutedEdges);

      // Reset selection state
      setSelectedNodeIds([]);
      setHighlightedEdgeIds([]);
      setPaths([]);

      addLog(
        "ok",
        `Graph generated: ${graphNodes.length} nodes, ${graphEdges.length} edges`
      );
    },
    [addLog]
  );

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      setSelectedNodeIds((prev) => {
        if (prev.includes(nodeId)) {
          addLog("info", `Node deselected: ${nodeId}`);
          return prev.filter((id) => id !== nodeId);
        } else {
          addLog("ok", `Node selected: ${nodeId}`);
          return [...prev, nodeId];
        }
      });
    },
    [addLog]
  );

  const handleDeselectNode = useCallback(
    (nodeId: string) => {
      setSelectedNodeIds((prev) => prev.filter((id) => id !== nodeId));
      addLog("info", `Node deselected: ${nodeId}`);
    },
    [addLog]
  );

  const handleClearSelection = useCallback(() => {
    setSelectedNodeIds([]);
    setHighlightedEdgeIds([]);
    setPaths([]);
    addLog("info", "Selection cleared");
  }, [addLog]);

  const handleFindPaths = useCallback(
    ({ directOnly, maxDepth }: { directOnly: boolean; maxDepth: number }) => {
      if (selectedNodeIds.length < 2) {
        addLog("warn", "Select at least 2 nodes");
        return;
      }

      setIsSearching(true);
      addLog("info", "Searching for paths...");

      // Simulate async for better UX
      setTimeout(() => {
        const foundPaths = findAllPaths(edges, selectedNodeIds, {
          directOnly,
          maxDepth,
        });
        setPaths(foundPaths);
        setIsSearching(false);

        if (foundPaths.length > 0) {
          addLog("ok", `${foundPaths.length} paths found`);
        } else {
          addLog("warn", "No paths found between selected nodes");
        }
      }, 300);
    },
    [edges, selectedNodeIds, addLog]
  );

  const handleHighlightPath = useCallback(
    (path: string[]) => {
      const edgeIds = getEdgeIdsFromPaths([path], edges);
      setHighlightedEdgeIds(edgeIds);
      addLog("info", `Highlighting path (${path.length - 1} edges)`);
    },
    [edges, addLog]
  );

  const handleHighlightAll = useCallback(() => {
    const edgeIds = getEdgeIdsFromPaths(paths, edges);
    setHighlightedEdgeIds(edgeIds);
    addLog("ok", `Highlighting all ${paths.length} paths`);
  }, [paths, edges, addLog]);

  const handleClearHighlights = useCallback(() => {
    setHighlightedEdgeIds([]);
    addLog("info", "Highlights cleared");
  }, [addLog]);

  const handleLoadSample = useCallback(async () => {
    try {
      const response = await fetch("/sample_matrix.csv");
      const text = await response.text();

      const lines = text.trim().split("\n");
      const headers = lines[0].split(",");
      const nodeNames = headers.slice(1);

      const matrix: number[][] = [];
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",").slice(1);
        matrix.push(values.map((v) => parseInt(v, 10) || 0));
      }

      setUploadStatus({
        success: true,
        message: `${nodeNames.length} nodes loaded`,
      });
      handleFileUpload({ nodeNames, matrix });
      addLog("ok", "Sample data loaded");
    } catch (error) {
      addLog("error", "Failed to load sample data");
    }
  }, [handleFileUpload, addLog]);

  const sidebarContent = (
    <>
      <UploadSection
        onFileUpload={handleFileUpload}
        uploadStatus={uploadStatus}
        setUploadStatus={setUploadStatus}
      />

      {/* Load Sample Data button */}
      <button onClick={handleLoadSample} className="btn-terminal w-full mt-4">
        [ LOAD SAMPLE DATA ]
      </button>

      <MatrixPreview
        nodeNames={nodeNames}
        matrix={matrix}
        isExpanded={isMatrixExpanded}
        onToggle={() => setIsMatrixExpanded(!isMatrixExpanded)}
      />

      <SelectedNodesPanel
        selectedNodes={selectedNodeIds}
        nodeNames={nodeNames}
        onDeselect={handleDeselectNode}
        onClearAll={handleClearSelection}
      />

      <PathFinderPanel
        selectedNodes={selectedNodeIds}
        nodeNames={nodeNames}
        paths={paths}
        onFindPaths={handleFindPaths}
        onHighlightPath={handleHighlightPath}
        onHighlightAll={handleHighlightAll}
        onClearHighlights={handleClearHighlights}
        isSearching={isSearching}
      />

      <StatusLog logs={logs} />
    </>
  );

  return (
    <div className="flex flex-col h-screen bg-terminal-black text-terminal-green font-mono overflow-hidden">
      <CRTOverlay />
      <Header />

      {isMobile ? (
        // Stacked layout for mobile
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="p-4 overflow-y-auto max-h-[50vh] border-b border-terminal-green-muted">
            {sidebarContent}
          </div>
          <div className="flex-1 min-h-[300px]">
            {nodes.length > 0 ? (
              <GraphCanvas
                initialNodes={nodes}
                initialEdges={edges}
                onNodeClick={handleNodeClick}
                selectedNodeIds={selectedNodeIds}
                highlightedEdgeIds={highlightedEdgeIds}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-green opacity-50">
                <p>// UPLOAD ADJACENCY MATRIX TO RENDER GRAPH</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        // Side-by-side for desktop
        <div className="flex flex-1 overflow-hidden">
          <aside className="w-80 overflow-y-auto border-r border-terminal-green-muted p-4 flex-shrink-0">
            {sidebarContent}
          </aside>
          <main className="flex-1 min-w-0">
            {nodes.length > 0 ? (
              <GraphCanvas
                initialNodes={nodes}
                initialEdges={edges}
                onNodeClick={handleNodeClick}
                selectedNodeIds={selectedNodeIds}
                highlightedEdgeIds={highlightedEdgeIds}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-terminal-green opacity-50">
                <p className="text-center">
                  // UPLOAD ADJACENCY MATRIX TO RENDER GRAPH
                  <br />
                  <span className="text-xs">
                    OR CLICK &quot;LOAD SAMPLE DATA&quot;
                  </span>
                </p>
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}
