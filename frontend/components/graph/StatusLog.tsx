"use client";

import { useRef, useEffect } from "react";

export interface LogEntry {
  timestamp: string;
  type: "ok" | "error" | "warn" | "info";
  message: string;
}

interface StatusLogProps {
  logs: LogEntry[];
}

export const StatusLog = ({ logs }: StatusLogProps) => {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  const getLogClassName = (type: LogEntry["type"]) => {
    switch (type) {
      case "ok":
        return "status-ok";
      case "error":
        return "status-err";
      case "warn":
        return "status-warn";
      default:
        return "text-terminal-green opacity-70";
    }
  };

  return (
    <div className="terminal-window mt-4">
      <div className="terminal-window-header">&gt; SYSTEM LOG</div>

      <div
        ref={logRef}
        className="h-32 overflow-y-auto font-mono text-xs space-y-1"
      >
        {logs.map((log, idx) => (
          <div key={idx} className={getLogClassName(log.type)}>
            [{log.timestamp}] {log.message}
          </div>
        ))}
      </div>
    </div>
  );
};

export const getCurrentTime = (): string => {
  const now = new Date();
  return now.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};
