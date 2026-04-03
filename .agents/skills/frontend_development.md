# Skill: Frontend Development & Vibe Coding

## Context
You are working on the "Audio Deep Learning Archive" project. This is a Jekyll static site on GitHub Pages.

## Architecture Rules
1. **Dynamic Pages go in `_pages/`:** If building a tool (like a generator or workspace), create an HTML file here with `layout: default` frontmatter.
2. **"WOW" Factor is Mandatory:** Never build plain, unstyled HTML. 
   - ALWAYS use classes defined in `_sass/premium-dashboard.scss`.
   - Use `class="glass-panel"`.
   - Incorporate `var(--accent-primary)` and `var(--accent-secondary)`.

## Workflow Rules (Vibe Coding)
When the user asks you to implement a new feature:
1. **Read the PRD First:** Start by calling `view_file` on `docs/01_prd_agentic_dashboard.md` to understand context.
2. **Update the PRD (If required):** If the user is proposing a new feature, update the PRD in `docs/` before you start writing code.
3. **Update Architecture (If required):** If you add a new folder or API integration, document it in `docs/02_architecture.md`.
4. **Implement:** Write the code adhering to the Aesthetic standards.
