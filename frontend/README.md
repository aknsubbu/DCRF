# DCRF Frontend - Causal Graph Visualizer

A terminal-themed web application for visualizing and exploring causal graphs. Upload adjacency matrices in CSV format and interactively explore node relationships, find paths between variables, and analyze graph structure.

## Features

- **CSV Upload** - Import adjacency matrices with drag-and-drop or file selection
- **Interactive Graph** - Pan, zoom, and click nodes using React Flow
- **Path Finding** - BFS-based algorithm to find all paths between selected nodes
- **Path Highlighting** - Visualize discovered paths with animated edge highlighting
- **Auto Layout** - Dagre-powered automatic graph layout (top-to-bottom)
- **Sample Data** - Built-in sample matrix for quick exploration
- **Terminal Theme** - Retro CRT-style UI with green-on-black aesthetic
- **Responsive** - Adapts to mobile and desktop viewports

## Tech Stack

- [Next.js 15](https://nextjs.org/) - React framework with App Router & Turbopack
- [React Flow](https://reactflow.dev/) (@xyflow/react) - Graph visualization
- [Dagre](https://github.com/dagrejs/dagre) - Graph layout algorithm
- [HeroUI v2](https://heroui.com/) - UI component library
- [Tailwind CSS 4](https://tailwindcss.com/) - Utility-first styling
- [TypeScript 5.6](https://www.typescriptlang.org/) - Type safety
- [Framer Motion](https://www.framer.com/motion/) - Animations

## Getting Started

### Prerequisites

- Node.js 18+
- npm, yarn, pnpm, or bun

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd frontend

# Install dependencies
npm install
```

### Development

```bash
# Start development server with Turbopack
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production

```bash
# Build for production
npm run build

# Start production server
npm run start
```

### Linting

```bash
# Run ESLint with auto-fix
npm run lint
```

## Usage

### CSV Format

Upload an adjacency matrix CSV file where:

- First row contains column headers (node names)
- First column contains row labels (node names)
- Cell values represent edge weights (0 = no edge, 1+ = edge exists)

Example:

```csv
,A,B,C
A,0,1,0
B,0,0,1
C,0,0,0
```

This creates edges: A → B → C

### Keyboard Shortcuts

- **Escape** - Clear all selections and highlights

### Workflow

1. Upload a CSV adjacency matrix or click "Load Sample Data"
2. Click nodes to select them (multi-select supported)
3. Use the Path Finder panel to search for paths between selected nodes
4. Click individual paths to highlight them, or "Highlight All"
5. View the status log for operation feedback

## Project Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # Root layout with providers
│   ├── page.tsx            # Main visualization page
│   └── providers.tsx       # HeroUI + Theme providers
├── components/
│   └── graph/              # Graph visualization components
│       ├── GraphCanvas.tsx # React Flow canvas wrapper
│       ├── TerminalNode.tsx# Custom styled node component
│       ├── UploadSection.tsx# CSV upload with dropzone
│       ├── PathFinderPanel.tsx# Path search controls
│       ├── SelectedNodesPanel.tsx# Selected nodes display
│       ├── MatrixPreview.tsx# Adjacency matrix viewer
│       ├── StatusLog.tsx   # System log component
│       ├── Header.tsx      # App header
│       └── CRTOverlay.tsx  # CRT visual effect
├── lib/                    # Utility functions
│   ├── graphUtils.ts       # Graph conversion & layout
│   └── pathFinding.ts      # BFS path algorithms
├── types/                  # TypeScript type definitions
├── styles/
│   └── globals.css         # Global styles & terminal theme
└── public/
    └── sample_matrix.csv   # Sample data file
```

## Contributing

See [AGENTS.md](./AGENTS.md) for code style guidelines and development instructions.

## License

MIT
