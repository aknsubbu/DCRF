"use client";

import { useState } from "react";
import { formatPathForDisplay } from "@/lib/pathFinding";

interface PathFinderPanelProps {
  selectedNodes: string[];
  nodeNames: string[];
  paths: string[][];
  onFindPaths: (options: { directOnly: boolean; maxDepth: number }) => void;
  onHighlightPath: (path: string[]) => void;
  onHighlightAll: () => void;
  onClearHighlights: () => void;
  isSearching: boolean;
}

export const PathFinderPanel = ({
  selectedNodes,
  nodeNames,
  paths,
  onFindPaths,
  onHighlightPath,
  onHighlightAll,
  onClearHighlights,
  isSearching,
}: PathFinderPanelProps) => {
  const [maxDepth, setMaxDepth] = useState(5);
  const [directOnly, setDirectOnly] = useState(false);

  const handleSearch = () => {
    onFindPaths({ directOnly, maxDepth });
  };

  return (
    <div className="terminal-window mt-4">
      <div className="terminal-window-header">&gt; PATHFINDER</div>

      {/* Configuration */}
      <div className="space-y-3 mb-4">
        <label className="flex items-center space-x-2 cursor-pointer group">
          <input
            type="checkbox"
            checked={directOnly}
            onChange={(e) => setDirectOnly(e.target.checked)}
            className="accent-terminal-green"
          />
          <span className="text-sm group-hover:text-terminal-amber transition-colors">
            DIRECT PATHS ONLY
          </span>
        </label>

        {!directOnly && (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-terminal-green opacity-70">
              MAX DEPTH:
            </span>
            <input
              type="number"
              value={maxDepth}
              onChange={(e) => setMaxDepth(parseInt(e.target.value, 10) || 5)}
              min="1"
              max="10"
              className="input-terminal w-16 text-center"
            />
          </div>
        )}

        <button
          onClick={handleSearch}
          disabled={selectedNodes.length < 2 || isSearching}
          className="btn-terminal w-full"
        >
          {isSearching ? "SEARCHING..." : "[ EXECUTE SEARCH ]"}
        </button>
      </div>

      {/* Divider */}
      <div className="border-b border-terminal-green-muted my-4 opacity-30" />

      {/* Results */}
      {paths.length > 0 ? (
        <>
          <div className="status-ok mb-4">{paths.length} PATHS FOUND</div>

          <div className="space-y-3 max-h-64 overflow-y-auto">
            {paths.map((path, idx) => (
              <div
                key={idx}
                className="border border-terminal-green-muted/30 p-2 hover:border-terminal-green transition-all group"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <span className="text-terminal-amber text-xs">
                      [{idx + 1}]
                    </span>
                    <span className="text-terminal-green text-sm ml-2 font-mono">
                      {formatPathForDisplay(path, nodeNames)}
                    </span>
                  </div>
                </div>
                <div className="mt-1 flex items-center justify-between text-xs">
                  <span className="text-terminal-green opacity-50">
                    LENGTH: {path.length - 1}
                  </span>
                  <button
                    onClick={() => onHighlightPath(path)}
                    className="text-terminal-amber hover:text-terminal-green transition-colors opacity-0 group-hover:opacity-100"
                  >
                    [HIGHLIGHT]
                  </button>
                </div>
              </div>
            ))}
          </div>

          <div className="flex space-x-2 mt-4">
            <button onClick={onHighlightAll} className="btn-terminal flex-1">
              [ HIGHLIGHT ALL ]
            </button>
            <button onClick={onClearHighlights} className="btn-terminal flex-1">
              [ CLEAR ]
            </button>
          </div>
        </>
      ) : (
        <p className="text-terminal-green opacity-50 text-sm">
          // SELECT 2+ NODES AND EXECUTE SEARCH
        </p>
      )}
    </div>
  );
};
