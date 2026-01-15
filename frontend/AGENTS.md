# AGENTS.md - Frontend Coding Agent Instructions

## Project Overview

**DCRF Frontend** - A Next.js 15 application for visualizing causal graphs. Provides interactive graph visualization with React Flow, CSV upload for adjacency matrices, and path-finding features.

**Tech Stack:** Next.js 15 (App Router, Turbopack), React 18, TypeScript 5.6, Tailwind CSS 4, HeroUI v2, React Flow (@xyflow/react), dagre

## Build/Run Commands

```bash
# Install dependencies
npm install

# Development server (with Turbopack)
npm run dev

# Production build
npm run build

# Start production server
npm run start

# Lint and auto-fix
npm run lint
```

### Testing

**No test framework configured.** Add tests with Vitest or Jest if needed:

```bash
npm install -D vitest @testing-library/react
```

## Code Style Guidelines

### File Naming

- **Components:** `PascalCase.tsx` (e.g., `GraphCanvas.tsx`, `TerminalNode.tsx`)
- **Utilities:** `camelCase.ts` (e.g., `graphUtils.ts`, `pathFinding.ts`)
- **Config files:** `camelCase.ts` (e.g., `site.ts`, `fonts.ts`)

### Import Order (ESLint enforced)

Imports must be grouped with blank lines between groups:

```typescript
// 1. Type imports
import type { Node, Edge } from "@xyflow/react";

// 2. React/built-in
import { useState, useCallback, useEffect } from "react";

// 3. External packages
import { ReactFlow, Controls, Background } from "@xyflow/react";
import clsx from "clsx";

// 4. Internal imports (use @/ alias)
import { Header } from "@/components/graph";
import { adjacencyMatrixToGraph } from "@/lib/graphUtils";
```

### TypeScript

- **Strict mode** is enabled - no implicit any
- Use `interface` for component props
- Use `type` for unions and utility types
- Separate `import type` for type-only imports
- Path alias: `@/*` maps to project root

```typescript
interface GraphCanvasProps {
  initialNodes: Node[];
  initialEdges: Edge[];
  onNodeClick: (nodeId: string) => void;
  selectedNodeIds: string[];
}

export const GraphCanvas = ({ initialNodes, ...props }: GraphCanvasProps) => {
  // ...
};
```

### React Patterns

- Add `"use client"` directive at top of client components
- Use **named exports** (not default exports)
- Use `memo()` for expensive components, set `displayName`
- Use `useCallback` for handlers passed as props
- Use barrel exports (`index.ts`) for component directories

```typescript
"use client";

import { memo, useCallback } from "react";

export const TerminalNode = memo(({ data, selected }: NodeProps) => {
  // ...
});
TerminalNode.displayName = "TerminalNode";
```

### JSX Props (ESLint enforced)

- Reserved props first (`key`, `ref`)
- Shorthand props next
- Alphabetical order
- Callbacks last

```tsx
<ReactFlow
  key={id}
  fitView
  attributionPosition="bottom-left"
  className="bg-terminal-black"
  edges={edges}
  nodes={nodes}
  onEdgesChange={onEdgesChange}
  onNodeClick={handleNodeClick}
/>
```

### Formatting Rules (ESLint/Prettier)

- Double quotes for strings
- Trailing commas in multiline
- Self-closing tags for childless elements
- Blank line before `return` statements
- Blank line after variable declarations

```typescript
const [nodes, setNodes] = useNodesState(initialNodes);
const [edges, setEdges] = useEdgesState(initialEdges);

useEffect(() => {
  setNodes(initialNodes);
}, [initialNodes]);

return <div>{/* ... */}</div>;
```

### Error Handling

```typescript
try {
  const result = parseCSV(file);
} catch (error) {
  setUploadStatus({
    success: false,
    message: error instanceof Error ? error.message : "Parse error",
  });
}
```

### Documentation

Use JSDoc for utility functions:

```typescript
/**
 * Convert adjacency matrix to React Flow nodes and edges
 */
export function adjacencyMatrixToGraph(
  nodeNames: string[],
  matrix: number[][],
): { nodes: Node[]; edges: Edge[] } {
  // ...
}
```

### CSS/Styling

- Use Tailwind CSS utility classes
- Use `clsx` for conditional classes
- Custom terminal theme colors defined in `globals.css`
- Use `tailwind-variants` for component variants

```typescript
import clsx from "clsx";

className={clsx(
  "border-2 px-4 py-3",
  selected ? "border-terminal-amber" : "border-terminal-green"
)}
```

## Project Structure

```
frontend/
├── app/                    # Next.js App Router pages
│   ├── layout.tsx          # Root layout with providers
│   ├── page.tsx            # Main graph visualization page
│   └── providers.tsx       # HeroUI + Theme providers
├── components/
│   └── graph/              # Graph visualization components
│       ├── index.ts        # Barrel exports
│       ├── GraphCanvas.tsx # React Flow canvas
│       ├── TerminalNode.tsx
│       └── UploadSection.tsx
├── lib/                    # Utility functions
│   ├── graphUtils.ts       # Graph conversion & layout
│   └── pathFinding.ts      # BFS path algorithms
├── types/
│   └── index.ts            # Shared TypeScript types
└── styles/
    └── globals.css         # Global styles + Tailwind
```

## Key Dependencies

| Package          | Purpose                          |
| ---------------- | -------------------------------- |
| `@xyflow/react`  | Graph visualization (React Flow) |
| `dagre`          | Graph layout algorithm           |
| `@heroui/react`  | UI component library             |
| `papaparse`      | CSV parsing                      |
| `react-dropzone` | File upload                      |
| `framer-motion`  | Animations                       |
| `clsx`           | Conditional class composition    |
