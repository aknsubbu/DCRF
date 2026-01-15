"use client";

import clsx from "clsx";

interface MatrixPreviewProps {
  nodeNames: string[];
  matrix: number[][];
  isExpanded: boolean;
  onToggle: () => void;
}

export const MatrixPreview = ({
  nodeNames,
  matrix,
  isExpanded,
  onToggle,
}: MatrixPreviewProps) => {
  if (!matrix.length) return null;

  return (
    <div className="terminal-window mt-4">
      <div
        className="terminal-window-header cursor-pointer flex justify-between items-center"
        onClick={onToggle}
      >
        <span>
          &gt; ADJACENCY MATRIX [{nodeNames.length}x{nodeNames.length}]
        </span>
        <span className="text-terminal-amber">
          {isExpanded ? "[-]" : "[+]"}
        </span>
      </div>

      {isExpanded && (
        <div className="overflow-x-auto">
          <table className="w-full font-mono text-xs border-collapse">
            <thead>
              <tr className="border-b border-terminal-green-muted">
                <th className="px-2 py-1 text-left text-terminal-green opacity-70">
                  NODE
                </th>
                {nodeNames.map((name, i) => (
                  <th
                    key={i}
                    className="px-2 py-1 text-terminal-green text-center"
                    title={name}
                  >
                    {name.substring(0, 3).toUpperCase()}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i} className="border-b border-terminal-green-muted/30">
                  <td
                    className="px-2 py-1 text-terminal-green opacity-70"
                    title={nodeNames[i]}
                  >
                    {nodeNames[i]?.substring(0, 8) || `node-${i}`}
                  </td>
                  {row.map((val, j) => (
                    <td
                      key={j}
                      className={clsx(
                        "px-2 py-1 text-center",
                        val === 0
                          ? "text-terminal-green opacity-30"
                          : "text-terminal-amber text-glow-amber"
                      )}
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
