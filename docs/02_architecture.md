# System Architecture: Agentic Dashboard

## Architecture Overview
The system relies on a **Static Site Generator (Jekyll)** hosted on **GitHub Pages**, augmented by **Client-Side Rendering (CSR)** and **Third-Party APIs** for dynamic behavior.

## Directory Structure & Responsibilities

```text
hshlalla.github.io/
├── .agents/            # Agent Configuration (The "HOW")
│   ├── skills/         # Specific instructions for AI agents
│   └── workflows/      # Automated multi-step processes
├── docs/               # Project Requirements (The "WHAT")
│   ├── 01_prd...md     # Product Requirements Document
│   └── 02_arch...md    # This file
├── front/              # Isolated frontend preview directory
│   └── index.html      # Raw HTML/CSS/JS work-in-progress (Previewed in /workspace)
├── _pages/             # Jekyll Pages (Dynamic Views)
│   ├── prd-generator.html # Parses MD files
│   └── workspace.html     # Fetches GitHub API & loads <iframe src="/front/">
├── _posts/             # Jekyll Blog Posts (Research/Portfolios)
├── _sass/              # Global Styles
│   ├── premium-dashboard.scss # Core glassmorphism logic
│   └── main.scss       # Import hub
└── index.html          # Main Agent Command Center Dashboard
```

## Data Flow Methods
Because this is a static site (`Server = GitHub Pages CDN`), direct database interactions are impossible. We solve this via:

1. **GitHub Issues API (Feedback)**
   - Client Types Feedback in `/workspace/` -> JS triggers `window.open("github.com/.../issues/new?title=...")` -> Developer handles Issue in GitHub.
2. **GitHub Content API (Live Status)**
   - `/workspace/` loads -> JS `fetch()` requests `api.github.com/repos/.../contents/docs` -> Renders list of latest PRDs.
3. **Local FileReader API (PRD Generator)**
   - Client uploads `.md` -> JS `FileReader` reads text -> Validates -> Renders HTML preview.
