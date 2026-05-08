---
name: gitnexus-setup
description: "Set up GitNexus MCP in Claude Code for a repository using official CLI commands. Examples: \"Set up GitNexus in this repo\", \"Add GitNexus to this project\", \"Configure GitNexus for Claude Code\""
---

# GitNexus Setup for Claude Code

Sets up GitNexus code intelligence in Claude Code using the official CLI. Two commands do everything: one registers the MCP globally with Claude Code, one indexes the repo and automatically installs skills, hooks, and context files.

## When to Use

- "Set up GitNexus in this repo"
- "Add GitNexus to this project"
- Onboarding a new project for Claude Code with code intelligence

## What Gets Installed

`npx gitnexus@latest analyze` handles all of this automatically:

| Item | Installed by |
|------|-------------|
| Code knowledge graph index (`.gitnexus/`) | `gitnexus analyze` |
| Agent skills (`.claude/skills/`) | `gitnexus analyze` |
| Claude Code hooks (PreToolUse + PostToolUse) | `gitnexus analyze` |
| `CLAUDE.md` context block | `gitnexus analyze` |
| MCP server (global, works for all repos) | `claude mcp add` (one-time) |

## Workflow

```
1. Run prereq check — git, Node.js, npm, OS (1 bash confirmation)
2. Register MCP globally via claude mcp add (1 bash confirmation, skip if already registered)
3. Run gitnexus analyze (1 bash confirmation)
4. Verify and tell user to restart Claude Code
```

Total bash confirmations: **3** (or **2** if MCP already registered).

---

## Step 1 — Prerequisites

Run this single script — **one bash confirmation for all prereqs:**

```bash
git rev-parse --show-toplevel 2>/dev/null || { echo "ERROR: not a git repo"; exit 1; }; \
node --version 2>/dev/null || { echo "ERROR: Node.js not found"; exit 2; }; \
npm --version 2>/dev/null || { echo "ERROR: npm not found"; exit 3; }; \
_os=$(uname -s 2>/dev/null); case "$_os" in MINGW*|MSYS*|CYGWIN*) echo "OS: Windows";; Darwin) echo "OS: Mac";; Linux) echo "OS: Linux";; *) echo "OS: Windows";; esac; \
echo "REPO: $(basename $(git rev-parse --show-toplevel))"; \
{ [ -f pom.xml ] || [ -f build.gradle ] || [ -f build.gradle.kts ]; } && echo "LANG: java"; \
{ [ -f requirements.txt ] || [ -f pyproject.toml ] || [ -f setup.py ]; } && echo "LANG: python"; \
[ -f go.mod ] && echo "LANG: go"; \
[ -f Cargo.toml ] && echo "LANG: rust"; \
[ -f package.json ] && echo "LANG: js/ts"; \
true
```

**Read the output:**

| Output | Action |
|--------|--------|
| `ERROR: not a git repo` | Offer `git init` — do not continue |
| `ERROR: Node.js not found` | **HARD STOP** — tell user to install Node.js 18+ |
| `ERROR: npm not found` | **HARD STOP** — npm ships with Node.js; tell user to reinstall Node.js as the installation is broken |
| `OS: Windows` | Use Windows commands in Steps 2 and 3 — and use **global install path** (see Step 3) instead of npx to avoid symlink EPERM errors |
| `OS: Mac` or `OS: Linux` | Use Mac/Linux commands in Steps 2 and 3 |
| `REPO: <name>` | Note for reference — `gitnexus analyze` creates `CLAUDE.md` automatically |
| `LANG: java` | Warn about build tools (see below) |

**Language support:**

| Language | Project file detected | Support | Action if detected |
|----------|--------------------|---------|-------------------|
| Python | `requirements.txt`, `pyproject.toml`, `setup.py` | Full | None |
| TypeScript / JavaScript | `package.json` | Full | None |
| Go | `go.mod` | Full | None |
| Rust | `Cargo.toml` | Full | None |
| Java | `pom.xml`, `build.gradle` | Requires build tools | Warn user |
| Other | — | May not be indexed | Inform user |

**If Java is detected**, warn before continuing:

> This repo contains Java files. GitNexus uses tree-sitter native parsers which require build tools.
> - **Windows**: Install Visual Studio Build Tools with the "Desktop development with C++" workload.
> - **Mac**: Run `xcode-select --install`.
> - **Linux**: Run `sudo apt install build-essential` (Debian/Ubuntu) or equivalent.
> Indexing may fail or produce 0 symbols if build tools are missing.

**Ensure the skill is under `.claude/skills/` before continuing.** Claude Code only loads skills from that path — a `gitnexus-setup` folder sitting at the repo root will be silently ignored and Claude may fall back to a copy from another repo. Run this as part of the prereq confirmation:

```bash
if [ -f ".claude/skills/gitnexus-setup/SKILL.md" ]; then
  echo "SKILL: already in correct location";
elif [ -f "gitnexus-setup/SKILL.md" ]; then
  mkdir -p ".claude/skills";
  mv "gitnexus-setup" ".claude/skills/gitnexus-setup";
  echo "SKILL: moved from root → .claude/skills/gitnexus-setup";
else
  mkdir -p ".claude/skills";
  echo "SKILLS DIR: .claude/skills/ created (skill already installed elsewhere or will be added by analyze)";
fi
```

---

## Step 2 — Register MCP (One-Time)

First, determine which Claude Code installation is in use:

- **Claude Code app installed** (standalone app or `claude` CLI available in terminal): use `claude mcp add` below.
- **Claude Code as VSCode/JetBrains extension only** (no `claude` CLI — running `claude` gives "not found" or "not a keyword"): skip to the **VSCode Extension Fallback** section below.

### Claude App / CLI path

Check if gitnexus is already registered:

```bash
claude mcp list
```

Look for `gitnexus` in the output. **If already registered:** skip to Step 3.

> If `claude mcp list` errors but the CLI exists, run `claude mcp add` directly — if already registered it will error and you can safely skip to Step 3.

**If not registered — Mac/Linux:**
```bash
claude mcp add gitnexus -- npx -y gitnexus@latest mcp
```

**If not registered — Windows:**
```bash
claude mcp add gitnexus -- cmd /c npx -y gitnexus@latest mcp
```

> This registers gitnexus globally with Claude Code. You only need to do this once — it works for all repos on this machine, regardless of which AI backend Claude Code uses (Anthropic direct, Azure AI Foundry, etc.).

### VSCode Extension Fallback — No `claude` CLI

When the `claude` CLI is unavailable, MCP servers are configured via a `.mcp.json` file in the repo root. Create or update it now.

**Mac/Linux — `.mcp.json`:**
```json
{
  "mcpServers": {
    "gitnexus": {
      "command": "npx",
      "args": ["-y", "gitnexus@latest", "mcp"]
    }
  }
}
```

**Windows — `.mcp.json`:**

Before writing `.mcp.json`, find the full path to the `gitnexus.cmd` binary:
```bash
npm prefix -g
```
This prints the npm global prefix (e.g. `C:\Users\<username>\AppData\Roaming\npm`). The gitnexus binary is at `<prefix>\gitnexus.cmd`.

Then write `.mcp.json` using that full path:
```json
{
  "mcpServers": {
    "gitnexus": {
      "command": "<full path to gitnexus.cmd>",
      "args": ["mcp"]
    }
  }
}
```

> **Windows note:** Use the full path to `gitnexus.cmd` rather than just `"gitnexus"` or `cmd /c gitnexus`. The npm global bin directory is frequently absent from the PATH seen by the VSCode extension process, causing the MCP server to fail silently. Using the absolute path bypasses this entirely.

If `.mcp.json` already exists with other servers, add the `"gitnexus"` key inside `"mcpServers"` — do not replace the whole file.

After writing `.mcp.json`, also ensure `settings.json` (`.claude/settings.json`) includes `"enabledMcpjsonServers": ["gitnexus"]` so the server is not blocked.

---

## Step 3 — Index the Repo

Run from the repo root — **one bash confirmation:**

**Mac/Linux:**
```bash
npx gitnexus@latest analyze
```

**Windows:**

On Windows, use a global install to avoid symlink EPERM errors that occur with npx and tree-sitter vendor packages.

First, check if gitnexus is already installed and what version is available:
```bash
npm list -g gitnexus 2>/dev/null && npm show gitnexus version
```

- If gitnexus is **not installed**, or the installed version is **behind** the latest: run the install + analyze together as one bash confirmation:
  ```bash
  npm install -g gitnexus@latest --install-strategy=nested && gitnexus analyze
  ```
- If the installed version **matches** latest: skip the install and run analyze directly:
  ```bash
  gitnexus analyze
  ```

> `--install-strategy=nested` tells npm to use folder copies instead of symlinks, bypassing the EPERM restriction on Windows machines without Developer Mode or admin privileges.
> After this, `gitnexus` is available as a global command — no npx needed for future runs.

This single command automatically:
- Builds the knowledge graph index (stored in `.gitnexus/`, gitignored)
- Installs 4 agent skills to `.claude/skills/`
- Registers Claude Code hooks (PreToolUse + PostToolUse) for auto-reindex
- Creates or updates `CLAUDE.md` with a GitNexus context block

> **Optional:** For semantic search (slower but better results), add `--embeddings` to the analyze command above.

> **If npx prompts to delete a corrupt cache:** approve the deletion. This only removes npx's own local package download cache (`_npx` folder inside the npm cache directory) — it does not touch your project files or data. The package will be re-downloaded automatically on the same run. Corruption typically happens from an interrupted previous download (network drop, antivirus interference, or a mid-write disk error).

> **If Java was detected and analyze fails:** build tools are missing — refer to the warning in Step 1.

---

## Step 4 — Verify and Tell the User

Run status to confirm the index:

**Mac/Linux:**
```bash
npx gitnexus@latest status
```

**Windows:**
```bash
gitnexus status
```

Then report to the user:

```
GitNexus is set up.

  ✓ MCP registered globally (claude mcp add)
  ✓ Index built: {N} symbols, {M} relationships
  ✓ Agent skills installed to .claude/skills/
  ✓ Claude Code hooks registered (PreToolUse + PostToolUse)
  ✓ CLAUDE.md updated with GitNexus context block

OS: {Windows/Mac/Linux}
Language(s) detected: {list}  [⚠ build tools required if Java]

ACTION REQUIRED: Restart Claude Code to activate the GitNexus MCP server.
After restarting, verify with: READ gitnexus://repo/{repo-name}/context
```

**If verification fails:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Server not found" / MCP tools missing | Claude Code not fully restarted | Restart Claude Code again |
| MCP server connects but read fails | MCP registration failed silently | Re-run Step 2 (`claude mcp add`) |
| Context loads but shows 0 symbols | No parseable source files, or build tools missing (Java) | Run `npx gitnexus@latest status` in terminal; install build tools if Java detected |

---

## Rollback — Undo Setup

| Item | How to undo |
|------|------------|
| MCP registration (CLI) | `claude mcp remove gitnexus` |
| MCP registration (VSCode) | Remove the `"gitnexus"` key from `.mcp.json`; remove `"gitnexus"` from `enabledMcpjsonServers` in `.claude/settings.json` |
| Global install (Windows) | `npm uninstall -g gitnexus` |
| Knowledge graph index | Delete `.gitnexus/` folder |
| Agent skills | Delete `.claude/skills/gitnexus*/` folders |
| `CLAUDE.md` block | Remove the `# GitNexus` section added by analyze |
| Hooks | Open `.claude/settings.json` — remove the gitnexus PostToolUse/PreToolUse entries |

---

## Completion Checklist

```
Bash confirmation 1 — prereq check:
- [ ] git repo confirmed
- [ ] Node.js present (HARD STOP if missing)
- [ ] npm present (HARD STOP if missing)
- [ ] OS captured — Windows/Mac/Linux
- [ ] Language(s) detected — build tools warning issued if Java

Bash confirmation 2 — MCP registration (skip if already registered):
- [ ] gitnexus MCP already registered, OR
- [ ] `claude` CLI available → registered via: `claude mcp add gitnexus -- [OS-appropriate command]`, OR
- [ ] `claude` CLI not available (VSCode extension) → `.mcp.json` written/updated with gitnexus entry; `enabledMcpjsonServers` confirmed in `.claude/settings.json`

Bash confirmation 3 — analyze + status:
- [ ] analyze completed successfully:
      Mac/Linux: `npx gitnexus@latest analyze`
      Windows: checked installed vs latest version (`npm list -g gitnexus` + `npm show gitnexus version`);
               installed/updated if needed (`npm install -g gitnexus@latest --install-strategy=nested`);
               then ran `gitnexus analyze`
- [ ] status confirms index (symbol count > 0):
      Mac/Linux: `npx gitnexus@latest status`
      Windows: `gitnexus status`

- [ ] User told to restart Claude Code
```
