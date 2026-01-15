"use client";

import { useEffect, useState } from "react";

export const Header = () => {
  const [currentTime, setCurrentTime] = useState<string>("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      );
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-terminal-green-muted bg-terminal-black px-6 py-3">
      <div className="flex justify-between items-center font-mono">
        <div className="text-glow-strong">
          <span className="text-terminal-green">+--- </span>
          <span className="uppercase tracking-wider text-terminal-green">
            CAUSAL GRAPH ANALYSIS SYSTEM
          </span>
          <span className="text-terminal-green"> v1.0.0 ---+</span>
        </div>
        <div className="text-terminal-green text-sm flex items-center gap-4">
          <span className="opacity-70">[</span>
          <span className="cursor" />
          <span className="opacity-70">]</span>
          <span className="font-mono">{currentTime}</span>
        </div>
      </div>
    </header>
  );
};
