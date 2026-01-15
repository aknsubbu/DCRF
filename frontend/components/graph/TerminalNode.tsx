"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import clsx from "clsx";

interface TerminalNodeData {
  label: string;
}

export const TerminalNode = memo(
  ({ data, selected }: NodeProps<{ data: TerminalNodeData }>) => {
    return (
      <div
        className={clsx(
          "border-2 px-4 py-3 font-mono uppercase text-xs tracking-wider min-w-[120px] text-center",
          selected
            ? "border-terminal-amber bg-terminal-amber text-terminal-black shadow-glow-amber"
            : "border-terminal-green bg-terminal-black text-terminal-green shadow-glow-green"
        )}
      >
        <Handle
          type="target"
          position={Position.Top}
          className="!w-2 !h-2 !bg-terminal-green !border-0"
        />

        <div className={clsx(selected ? "" : "text-glow")}>
          {(data as TerminalNodeData).label}
          {selected && <span className="cursor ml-1" />}
        </div>

        <Handle
          type="source"
          position={Position.Bottom}
          className="!w-2 !h-2 !bg-terminal-green !border-0"
        />
      </div>
    );
  }
);

TerminalNode.displayName = "TerminalNode";
