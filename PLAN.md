# Interactive Neural Network Visualization

## Goal
Build a web-based interactive tool to step through neural network training, showing:
- Network structure with weights and activations
- Forward pass algebra step-by-step
- Backward pass (gradient) algebra step-by-step
- How weights change after each update

## Architecture Decision

**Option A: Pure TypeScript (Recommended)**
- Reimplement the neural network in TypeScript
- Everything runs in browser, no backend needed
- Reinforces understanding by coding it again
- Easy to deploy (static site)

**Option B: Python Backend + React Frontend**
- Reuse existing Python code
- More complex (needs API layer)
- Harder to deploy

**Recommendation**: Option A - pure TypeScript

## Tech Stack
- React 18 + TypeScript
- Vite for build tooling
- TailwindCSS for styling
- SVG for network rendering (more flexible than Canvas for interactivity)
- KaTeX or MathJax for rendering math equations

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Training Controls: [◀ Prev] [Step ▶] [Play ▶▶] [Reset ↺]      │
│  Step: 0/2000  |  Input: [0,1]  |  Target: 1  |  Loss: 0.693   │
├───────────────────────────────────┬─────────────────────────────┤
│                                   │                             │
│      Network Visualization        │    Computation Panel        │
│                                   │                             │
│    (in0)──w00──►(h0)──w00──►      │    Forward Pass:            │
│       ╲        ╱    ╲             │    z₀ = x₀·w₀₀ + x₁·w₁₀ + b │
│        ╲      ╱      ╲            │       = 0·1.2 + 1·0.8 + 0.1 │
│         ╲    ╱        ►(out)      │       = 0.9                  │
│          ╲  ╱        ╱            │    a₀ = ReLU(0.9) = 0.9     │
│           ╲╱        ╱             │    ...                       │
│    (in1)──w10──►(h1)──w10──►      │                             │
│                                   │    Backward Pass:           │
│   [Weights shown on edges]        │    ∂L/∂p = -y/p + (1-y)/(1-p)│
│   [Activations shown in nodes]    │         = -1/0.7 + 0        │
│   [Gradients shown on hover]      │         = -1.43             │
│                                   │    ...                       │
├───────────────────────────────────┴─────────────────────────────┤
│  Data: ○[0,0]→0  ●[0,1]→1  ○[1,0]→1  ○[1,1]→0                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Core Components
1. `NetworkVisualization` - SVG-based network diagram
   - Nodes for inputs, hidden, output
   - Edges with weight labels
   - Color coding (blue=positive, red=negative)
   - Hover to see gradients

2. `ComputationPanel` - Shows math step-by-step
   - Forward pass section
   - Backward pass section
   - Highlight current computation
   - Use KaTeX for proper math rendering

3. `TrainingControls` - Navigation
   - Step forward/back
   - Play animation
   - Reset to initial state
   - Speed control

4. `DataSelector` - Choose which input to visualize
   - Show all 4 XOR cases
   - Highlight current selection

### State Management
```typescript
interface NetworkState {
  weights: number[][][];      // weights[layer][from][to]
  biases: number[][];         // biases[layer][node]
  // Cached from forward pass
  preActivations: number[][]; // z values before ReLU
  activations: number[][];    // values after ReLU
  // Cached from backward pass
  weightGrads: number[][][];
  biasGrads: number[][];
}

interface TrainingState {
  step: number;
  currentInput: number[];
  currentTarget: number;
  loss: number;
  history: NetworkState[];  // For stepping backward
}
```

## Implementation Steps

### Phase 1: Project Setup
1. Create React + TypeScript + Vite project
2. Add TailwindCSS
3. Add KaTeX for math rendering
4. Basic layout scaffolding

### Phase 2: Network Logic (TypeScript)
1. Implement `Network` class
   - Constructor with layer sizes
   - Forward pass (store intermediates)
   - Backward pass (compute gradients)
   - Step (apply gradients)
2. Port sigmoid, ReLU, BCE functions
3. Add tests (vitest)

### Phase 3: Network Visualization
1. SVG component for network structure
2. Draw nodes and edges
3. Add weight labels
4. Color coding by sign/magnitude
5. Show activations in nodes

### Phase 4: Computation Panel
1. Forward pass display
   - Show each layer's computation
   - Highlight current values
2. Backward pass display
   - Show gradient chain rule
   - Show each weight's gradient

### Phase 5: Interactivity
1. Step controls
2. History for backward stepping
3. Input selector
4. Play/pause animation

### Phase 6: Polish
1. Tooltips and explanations
2. Responsive design
3. Persist settings

## File Structure
```
web/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── network/
│   │   ├── Network.ts        # Neural network implementation
│   │   ├── Network.test.ts   # Tests
│   │   └── functions.ts      # sigmoid, relu, bce
│   ├── components/
│   │   ├── NetworkVisualization.tsx
│   │   ├── ComputationPanel.tsx
│   │   ├── TrainingControls.tsx
│   │   └── DataSelector.tsx
│   ├── hooks/
│   │   └── useTraining.ts    # Training state management
│   └── styles/
│       └── index.css
└── tests/
    └── ...
```

## Open Questions
1. Should we support different network architectures or hardcode XOR?
2. Should computation panel show all steps at once or animate through?
3. Do we want to save/load network states?
