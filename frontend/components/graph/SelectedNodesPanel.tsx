"use client";

interface SelectedNodesPanelProps {
  selectedNodes: string[];
  nodeNames: string[];
  onDeselect: (nodeId: string) => void;
  onClearAll: () => void;
}

export const SelectedNodesPanel = ({
  selectedNodes,
  nodeNames,
  onDeselect,
  onClearAll,
}: SelectedNodesPanelProps) => {
  return (
    <div className="terminal-window mt-4">
      <div className="terminal-window-header">
        &gt; SELECTED NODES [{selectedNodes.length}]
      </div>

      {selectedNodes.length === 0 ? (
        <p className="text-terminal-green opacity-50 text-sm">
          // NO NODES SELECTED
        </p>
      ) : (
        <div className="space-y-2">
          {selectedNodes.map((nodeId) => {
            const index = parseInt(nodeId.split("-")[1], 10);
            const nodeName = nodeNames[index] || nodeId;

            return (
              <div
                key={nodeId}
                className="flex items-center justify-between group hover:bg-terminal-green/10 p-1"
              >
                <span className="text-terminal-green font-mono text-sm">
                  [x] {nodeId}{" "}
                  <span className="text-terminal-amber">{nodeName}</span>
                </span>
                <button
                  onClick={() => onDeselect(nodeId)}
                  className="text-terminal-red opacity-0 group-hover:opacity-100 transition-opacity hover:text-glow"
                >
                  [X]
                </button>
              </div>
            );
          })}

          <button onClick={onClearAll} className="btn-terminal w-full mt-4">
            [ CLEAR ALL ]
          </button>
        </div>
      )}
    </div>
  );
};
