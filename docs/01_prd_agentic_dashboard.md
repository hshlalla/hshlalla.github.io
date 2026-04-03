# Product Requirements Document (PRD): Agentic Dashboard & Workspace

**Version:** 1.0 (Initial Foundation)
**Project Name:** Audio Deep Learning Archive Agent Workspace
**Platform:** GitHub Pages (Static hosting) with Client-side JS

## 1. Introduction
This project aims to transform a traditional personal portfolio site (Jekyll/Minimal Mistakes) into an "Agentic Workspace"—a dual-purpose platform serving both as a public-facing blog/portfolio and an interactive, real-time workspace for client and AI collaboration.

## 2. Target Audience
1. **The Owner (Developer/AI Engineer):** Needs a central hub to manage AI tasks, showcase research, and receive structured client feedback.
2. **Clients / Collaborators:** Need a simple way to view progress (frontend previews) and submit actionable feedback.
3. **AI Agents (e.g., Cursor, OpenClaw):** Need to be able to read system state, understand styling constraints, and execute workflows autonomously.

## 3. Core Features & Requirements

### 3.1 Public Portfolio (Foundation)
*   **Requirement:** Standard blog functionality for AI and Sound research.
*   **Implementation:** Maintained via Jekyll `_posts/` and `_pages/`.

### 3.2 Live Workspace (`/workspace/`)
*   **Requirement:** A dashboard for client review. Supports dynamic routing for multiple projects via URL parameter (`?project=name`).
*   **Features:**
    *   **Live Preview:** `<iframe src="/work/{project-name}/front/">` embedding the specific project's UI.
    *   **Status Tracking (Docs):** Fetch PRDs from the `/work/{project-name}/docs/` folder via GitHub REST API and render Mermaid diagrams visually.
    *   **Feedback Mechanism:** A text form that securely redirects user input into a pre-filled GitHub Issue URL (`/issues/new?title=...&body=...`), including the project label, ensuring all feedback is captured in the developer's formal tracking system.

### 3.3 MD to PRD Generator (`/prd-generator/`)
*   **Requirement:** A tool to rapid-prototype PRDs directly in the browser.
*   **Features:**
    *   Client-side file upload for `.md` files.
    *   *(Phase 2 API)* Send parsed markdown to an LLM (e.g., OpenAI API) connected via a serverless function, returning structured PRD HTML.

### 3.4 Agent Command Center (Dashboard UI)
*   **Requirement:** A central hub (`index.html`) to trigger automated workflows.
*   **Features:** Visual buttons linked to specific pages or automated AI instructions (e.g., `/ai_times_summary`, `/document_upload`).

## 4. Aesthetic Design (The "WOW" Factor)
*   **Theme:** "Premium Agentic Dashboard". Deep backgrounds (`#0a0a0c`), Glassmorphism panels (blur, semi-transparent borders).
*   **Colors:** Neon Purple (`#bc13fe`) and Cyan (`#00f2ff`) gradients.
*   **Animations:** Smooth hover states, glowing "alive" indicators (pulse animations), fade-in transitions.

## 5. Non-Functional Requirements
*   **Serverless constraint:** Since hosted on GitHub Pages, backend logic must rely on external REST APIs (like GitHub's) or purely client-side browser APIs.
*   **Responsiveness:** All features must work on Desktop and Mobile.
