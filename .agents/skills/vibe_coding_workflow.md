# Skill: Vibe Coding & PRD Management

## Context
You are working within the "Audio Deep Learning Archive" workspace.
This repository supports a multi-project Vibe Coding architecture located in the `work/` directory.

## Core Lifecycle
1. **Read Request & PRD:** When the user makes a request or creates a GitHub Issue, locate the relevant `work/{project-name}/docs/prd.md`.
2. **Update the PRD (The Contract):** If the user is asking for a new feature or structural change, YOU MUST UPDATE THE `prd.md` FIRST before writing any executable code.
3. **Update Architecture (Mermaid):** Ensure the Mermaid diagrams in `docs/arch.md` or `prd.md` accurately reflect the new data flow or component structure.
4. **Implement Code:** Write the code in `front/`, `back/`, or `ai/` adhering to the PRD.
5. **Commit & Sync:** The user will review the changes inside `hshlalla.github.io/workspace/?project={project-name}` and provide further feedback.

## Folder Structure Rules
Never place project-specific logic in the root Jekyll directories (`_pages`, `_posts`) unless it modifies the global dashboard itself.
ALL project work must remain strictly inside `work/{project-name}/`.

- `project.json`: Metadata config. Specifies if the project is `internal` (hosted here) or `external` (hosted on Vercel/Render).
- `front/`: Client-side HTML/JS/CSS (served by GitHub pages). Skip if `external`.
- `back/`: Server-side API logic. Skip if `external`.
- `ai/`: Prompts, RAG configurations, model bindings. Skip if `external`.
- `docs/`: The living PRD and Mermaid diagrams. This must ALWAYS be present locally, even for `external` projects. This repository is the Control Center.

## External Projects (Scalability)
If the project is massive (requiring Python backends, custom servers, etc.), you must create an entirely new GitHub repository for it.
However, you MUST still create a `work/{project-name}/` folder in this repository.
Inside it, place the `docs/prd.md` and a `project.json` like this:
```json
{
  "name": "Massive AI App",
  "description": "Hosted externally on Vercel",
  "preview_url": "https://external-url.app",
  "type": "external"
}
```
This ensures the client can still track PRD progress and view the live app via `hshlalla.github.io/workspace/?project={project-name}`.
