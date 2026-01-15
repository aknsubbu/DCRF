# Causal Graph Visualization Platform - Walkthrough

A **terminal-themed causal graph visualization web app** built with Next.js 15, React Flow, and Tailwind CSS.

---

## Final Result

![Causal Graph App](/Users/anandkumarns/.gemini/antigravity/brain/40af9a0b-4880-4b54-b852-8b82aa83946a/screenshot_final_state.png)

---

## Features Implemented

| Feature | Status |
|---------|--------|
| CSV Adjacency Matrix Upload | ✅ |
| Interactive Causal Graph | ✅ |
| Multi-Node Selection | ✅ |
| BFS Path Finding | ✅ |
| Path Edge Highlighting | ✅ |
| Direct/Indirect Path Filtering | ✅ |
| Terminal CLI Aesthetic | ✅ |
| CRT Scanline Effect | ✅ |
| Responsive Layout | ✅ |

---

## Files Created

### Core Components
- [CRTOverlay.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/CRTOverlay.tsx) - Scanline overlay effect
- [Header.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/Header.tsx) - Terminal header with clock
- [StatusLog.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/StatusLog.tsx) - Live status log panel
- [UploadSection.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/UploadSection.tsx) - CSV drag-drop upload
- [MatrixPreview.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/MatrixPreview.tsx) - Adjacency matrix table
- [GraphCanvas.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/GraphCanvas.tsx) - React Flow graph canvas
- [TerminalNode.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/TerminalNode.tsx) - Custom graph node
- [SelectedNodesPanel.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/SelectedNodesPanel.tsx) - Selection display
- [PathFinderPanel.tsx](file:///Volumes/DevDrive/DCRF/frontend/components/graph/PathFinderPanel.tsx) - Path search UI

### Utilities
- [graphUtils.ts](file:///Volumes/DevDrive/DCRF/frontend/lib/graphUtils.ts) - Matrix parsing, Dagre layout
- [pathFinding.ts](file:///Volumes/DevDrive/DCRF/frontend/lib/pathFinding.ts) - BFS algorithms

### Modified Files
- [globals.css](file:///Volumes/DevDrive/DCRF/frontend/styles/globals.css) - Terminal design system
- [page.tsx](file:///Volumes/DevDrive/DCRF/frontend/app/page.tsx) - Main app with state management
- [layout.tsx](file:///Volumes/DevDrive/DCRF/frontend/app/layout.tsx) - Simplified layout

---

## How to Use

1. **Start the server**: `npm run dev` in `/Volumes/DevDrive/DCRF/frontend`
2. **Load data**: Click "LOAD SAMPLE DATA" or upload a CSV adjacency matrix
3. **Select nodes**: Click nodes in the graph to select them
4. **Find paths**: Click "EXECUTE SEARCH" to find all paths between selected nodes
5. **Highlight**: Click "HIGHLIGHT" on individual paths or "HIGHLIGHT ALL"
6. **Clear**: Press `Escape` to clear selection, or use panel buttons

---

## Browser Recording

![Demo Recording](/Users/anandkumarns/.gemini/antigravity/brain/40af9a0b-4880-4b54-b852-8b82aa83946a/causal_graph_test_1768454009813.webp)
