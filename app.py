import asyncio
import json
import os
import re
import shutil
import subprocess
import urllib.parse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
PMS_DIR = DATA_DIR / "pms"
CONFIG_FILE = DATA_DIR / "config.json"

for folder in (REPOS_DIR, PMS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DEFAULT_VAULT_PATH = str(Path.home() / "Desktop" / "Product-Brain-Vault")

DEFAULT_GRAPHIFY_IGNORE = """# Skip test suites, fixtures, and generated type/bundle files for faster, cleaner PM graphs
tests/
test/
__tests__/
fixtures/
*.test.*
*.spec.*
*.d.ts
*.d.mts
*.min.js
dist/
build/
vendor/
node_modules/
"""

_last_vault_snapshot: Dict[str, Tuple[float, int]] = {}


class VaultConfigRequest(BaseModel):
    vault_path: str


class RepoRequest(BaseModel):
    repo_url: str
    github_token: Optional[str] = None


class NoteSaveRequest(BaseModel):
    rel_path: str
    content: str


class OpenNoteRequest(BaseModel):
    rel_path: Optional[str] = None


class ClaudeAskRequest(BaseModel):
    prompt: str
    mode: str = "ask"  # "ask", "overlap", "path", or "prd"
    selected_repos: List[str] = []
    selected_pms: List[str] = []
    session_id: Optional[str] = None


def get_vault_dir() -> Path:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            raw = data.get("vault_path", "").strip()
            if raw:
                return Path(raw).expanduser().resolve()
        except Exception:
            pass
    return Path(DEFAULT_VAULT_PATH).expanduser().resolve()


def set_vault_dir(raw_path: str) -> Path:
    vault = Path(raw_path.strip()).expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"vault_path": str(vault)}, indent=2), encoding="utf-8")
    return vault


def find_claude_binary() -> Optional[str]:
    found = shutil.which("claude")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / "claude",
        Path("/usr/local/bin/claude"),
        Path("/opt/homebrew/bin/claude"),
        Path.home() / ".npm-global" / "bin" / "claude",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, allow_codes=(0,), timeout: int = 180) -> str:
    env = os.environ.copy()
    extra_paths = [
        str(Path.home() / ".local" / "bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    env["PATH"] = ":".join(extra_paths + [env.get("PATH", "")])

    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if result.returncode not in allow_codes:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd[:4])}): {result.stderr or result.stdout}"
        )
    return (result.stdout or "") + (result.stderr or "")


def sanitize_name(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw.strip()).strip("-")


def resolve_vault_file(rel_path: str) -> Path:
    vault = get_vault_dir()
    target = (vault / rel_path).resolve()
    if vault not in target.parents and target != vault:
        raise HTTPException(status_code=400, detail="Invalid path outside Obsidian vault")
    return target


def write_claude_settings(vault: Path) -> Path:
    """Ensure Claude Code has explicit allow permissions for Graphify queries and PRD file writes."""
    claude_dir = vault / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.local.json"
    settings_payload = {
        "permissions": {
            "allow": [
                "Read",
                "Grep",
                "Glob",
                "Write",
                "Edit",
                "Bash(graphify *)",
            ]
        }
    }
    settings_path.write_text(json.dumps(settings_payload, indent=2), encoding="utf-8")
    return settings_path


def build_pm_doc_graph(pm_name: str, extra_dirs: Optional[List[Path]] = None) -> Path:
    """Convert a PM's PRDs and Obsidian notes into a Graphify-compatible graph.json."""
    pm_root = PMS_DIR / pm_name
    docs_dir = pm_root / "docs"
    out_dir = pm_root / "graphify-out"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = []
    edges = []
    seen_nodes = set()

    def add_node(node_id: str, label: str, source_file: str, community: str):
        if node_id not in seen_nodes:
            seen_nodes.add(node_id)
            nodes.append({
                "id": node_id,
                "label": label,
                "source_file": source_file,
                "community": community,
            })

    pm_node_id = f"pm:{pm_name}"
    add_node(pm_node_id, f"PM: {pm_name}", "workspace", f"pm_{pm_name}")

    scan_folders = [docs_dir] + (extra_dirs or [])
    seen_files = set()

    for folder in scan_folders:
        if not folder.exists():
            continue
        for doc_file in sorted(folder.glob("*.md")) + sorted(folder.glob("*.txt")):
            if not doc_file.is_file() or doc_file.name in seen_files:
                continue
            if doc_file.name.startswith("00-") or doc_file.name == "graph.canvas":
                continue
            seen_files.add(doc_file.name)

            text = doc_file.read_text(encoding="utf-8", errors="ignore")
            doc_id = f"prd:{pm_name}:{doc_file.name}"
            add_node(doc_id, f"{doc_file.name} (by {pm_name})", doc_file.name, f"pm_{pm_name}")
            edges.append({
                "source": pm_node_id,
                "target": doc_id,
                "relation": "authored",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
            })

            for line in text.splitlines():
                heading = re.match(r"^#{1,3}\s+(.+)$", line.strip())
                if heading:
                    topic = heading.group(1).strip()
                    topic_id = f"concept:{sanitize_name(topic).lower()}"
                    add_node(topic_id, topic, doc_file.name, f"pm_{pm_name}")
                    edges.append({
                        "source": doc_id,
                        "target": topic_id,
                        "relation": "specifies",
                        "confidence": "EXTRACTED",
                        "confidence_score": 1.0,
                    })

                for match in re.findall(
                    r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]|`([a-zA-Z0-9_.-]+)`|\*\*([a-zA-Z0-9 _.-]{3,40})\*\*",
                    line,
                ):
                    clean_term = (match[0] or match[1] or match[2]).strip()
                    if clean_term:
                        term_id = f"concept:{sanitize_name(clean_term).lower()}"
                        add_node(term_id, clean_term, doc_file.name, f"pm_{pm_name}")
                        edges.append({
                            "source": doc_id,
                            "target": term_id,
                            "relation": "references",
                            "confidence": "INFERRED",
                            "confidence_score": 0.85,
                        })

    graph_path = out_dir / "graph.json"
    graph_payload = {
        "directed": True,
        "multigraph": False,
        "graph": {"schema_version": "1.0", "source": f"pm_{pm_name}"},
        "nodes": nodes,
        "edges": edges,
        "links": edges,
    }
    graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    return graph_path


def scan_vault_human_notes(vault: Path) -> Dict[str, Tuple[float, int]]:
    snapshot: Dict[str, Tuple[float, int]] = {}
    for folder_name in ("PMs", "PRDs"):
        folder = vault / folder_name
        if not folder.exists():
            continue
        for f in folder.rglob("*.md"):
            if f.name.startswith("00-"):
                continue
            stat = f.stat()
            snapshot[str(f.relative_to(vault))] = (stat.st_mtime, stat.st_size)
    return snapshot


def sync_all_to_obsidian() -> dict:
    """Export all mapped GitHub repos, PM notes, and the unified global graph into the Obsidian vault."""
    global _last_vault_snapshot
    vault = get_vault_dir()
    repos_vault = vault / "Repos"
    pms_vault = vault / "PMs"
    prds_vault = vault / "PRDs"
    private_vault = vault / "Private"
    global_vault = vault / "Global-Graph"
    meta_dir = vault / "_graphify"

    for folder in (vault, repos_vault, pms_vault, prds_vault, private_vault, global_vault, meta_dir):
        folder.mkdir(parents=True, exist_ok=True)

    write_claude_settings(vault)

    private_readme = private_vault / "README.md"
    if not private_readme.exists():
        private_readme.write_text(
            "# Private Scratchpad\n\nNotes inside `Private/` stay private and are excluded from the shared Global Knowledge Graph until you move them into `PMs/` or `PRDs/`.\n",
            encoding="utf-8",
        )

    all_graph_paths: List[Path] = []
    synced_repo_names: List[str] = []
    synced_pm_names: List[str] = []

    # 1. Export each mapped GitHub repo's knowledge graph into Vault/Repos/<repo_name>/
    for repo_dir in sorted(REPOS_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        out_dir = repo_dir / "graphify-out"
        graph_json = out_dir / "graph.json"
        if not graph_json.exists():
            continue

        repo_name = repo_dir.name
        synced_repo_names.append(repo_name)
        all_graph_paths.append(graph_json)

        target_repo_vault = repos_vault / repo_name
        target_repo_vault.mkdir(parents=True, exist_ok=True)
        run_cmd(["graphify", "export", "obsidian", "--graph", str(graph_json), "--dir", str(target_repo_vault)])

        report_src = out_dir / "GRAPH_REPORT.md"
        if report_src.exists():
            shutil.copyfile(report_src, target_repo_vault / "00-Architecture-Report.md")

    # 2. Index PM folders (both uploaded docs and notes written directly inside Vault/PMs/<name>/)
    pm_names = set()
    for pm_dir in PMS_DIR.iterdir():
        if pm_dir.is_dir():
            pm_names.add(pm_dir.name)
    for vault_pm_dir in pms_vault.iterdir():
        if vault_pm_dir.is_dir():
            pm_names.add(vault_pm_dir.name)

    for pm_name in sorted(pm_names):
        target_pm_vault = pms_vault / pm_name
        target_pm_vault.mkdir(parents=True, exist_ok=True)
        graph_json = build_pm_doc_graph(pm_name, extra_dirs=[target_pm_vault])

        synced_pm_names.append(pm_name)
        all_graph_paths.append(graph_json)

        docs_dir = PMS_DIR / pm_name / "docs"
        if docs_dir.exists():
            for doc in docs_dir.glob("*"):
                if doc.is_file():
                    shutil.copyfile(doc, target_pm_vault / doc.name)

    # Also index any PRDs inside Vault/PRDs/
    if any(prds_vault.glob("*.md")):
        prds_graph = build_pm_doc_graph("Team-PRDs", extra_dirs=[prds_vault])
        all_graph_paths.append(prds_graph)

    # 3. Build unified Global Graph + Obsidian export across all Repos & PMs
    global_graph_json = meta_dir / "global_graph.json"
    if len(all_graph_paths) == 1:
        shutil.copyfile(all_graph_paths[0], global_graph_json)
    elif len(all_graph_paths) > 1:
        run_cmd(["graphify", "merge-graphs", *[str(p) for p in all_graph_paths], "--out", str(global_graph_json)])

    total_nodes = 0
    total_edges = 0
    if global_graph_json.exists():
        run_cmd(["graphify", "export", "obsidian", "--graph", str(global_graph_json), "--dir", str(global_vault)])
        try:
            g_data = json.loads(global_graph_json.read_text(encoding="utf-8"))
            total_nodes = len(g_data.get("nodes", []))
            total_edges = len(g_data.get("edges", g_data.get("links", [])))
        except Exception:
            pass

    # 4. Write Obsidian Home Map (00-Home.md) & Claude Code Instructions (CLAUDE.md)
    repo_links = "\n".join([f"- [[Repos/{r}/00-Architecture-Report|{r}]]" for r in synced_repo_names]) or "- _No repositories synced yet_"
    pm_links = "\n".join([f"- **PM: {p}** (`PMs/{p}/`)" for p in synced_pm_names]) or "- _No peer PM PRDs uploaded yet_"
    prd_links = "\n".join([f"- [[PRDs/{f.stem}|{f.name}]]" for f in sorted(prds_vault.glob("*.md"))]) or "- _No drafted PRDs yet_"

    home_md = f"""# Product Brain — Global Knowledge Graph

Welcome to **Product Brain**. Every connected GitHub repository and Product Manager spec is automatically mapped and linked here.
Press **`Cmd + G`** in Obsidian to inspect the interactive graph, or use **Claude Code** inside Product Brain to ask cross-repo questions and draft grounded PRDs.

## Synced GitHub Repositories
{repo_links}

## Peer PM Workspaces
{pm_links}

## Drafted PRDs
{prd_links}

## Global Cross-Repo Map
- Spatial Canvas: `Global-Graph/graph.canvas`
- Machine Graph: `_graphify/global_graph.json` ({total_nodes} nodes, {total_edges} edges)
"""
    (vault / "00-Home.md").write_text(home_md, encoding="utf-8")

    claude_md = """# CLAUDE.md — Product Brain Obsidian Vault

You are **Product Brain**, an AI Product Management Copilot operating inside an Obsidian vault powered by Graphify.

## Vault Layout
- `Repos/<repo-name>/`: Wikilinked Obsidian notes, `graph.canvas`, and `00-Architecture-Report.md` for each synced GitHub repository.
- `PMs/<pm-name>/`: Specifications and PRDs authored by peer Product Managers.
- `PRDs/`: Newly drafted PRDs generated by Product Brain.
- `Private/`: Local scratchpad notes — NEVER include or cite files from `Private/` unless the user explicitly asks about their own private scratchpad.
- `_graphify/global_graph.json`: Merged NetworkX knowledge graph spanning all repositories and PM workspaces.

## Operating Rules for Answering & Drafting PRDs
1. **Query the Knowledge Graph First:** Run Graphify CLI commands via Bash against `_graphify/global_graph.json` (or a scoped graph if provided in the prompt) before reading raw files:
   - `graphify query "<question>" --graph _graphify/global_graph.json --budget 3000`
   - `graphify path "<EntityA>" "<EntityB>" --graph _graphify/global_graph.json`
   - `graphify explain "<ComponentName>" --graph _graphify/global_graph.json`
2. **Cross-Reference Peer PM Knowledge:** Check `PMs/` and `PRDs/` to see if another PM has already scoped related features, constraints, or APIs.
3. **Ground Every Claim:** Clearly tag facts as **`[Code Verified]`** (citing exact repo name & file path) or **`[Peer PRD]`** (citing PM name & filename).
4. **Writing PRDs:** When asked to draft a PRD, write a clean, structured Markdown file directly into `PRDs/<YYYY-MM-DD-feature-slug>.md` with Obsidian `[[wikilinks]]` to the referenced services and peer PRDs, then summarize what you created and include the exact relative path `PRDs/<filename>.md`.
"""
    (vault / "CLAUDE.md").write_text(claude_md, encoding="utf-8")
    _last_vault_snapshot = scan_vault_human_notes(vault)

    return {
        "vault_path": str(vault),
        "repos_synced": len(synced_repo_names),
        "pms_synced": len(synced_pm_names),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


async def vault_watcher_loop():
    """Background watcher that re-indexes the global graph whenever you edit/create notes in Obsidian."""
    global _last_vault_snapshot
    while True:
        await asyncio.sleep(3)
        try:
            vault = get_vault_dir()
            if not vault.exists():
                continue
            current = scan_vault_human_notes(vault)
            if current != _last_vault_snapshot:
                await asyncio.to_thread(sync_all_to_obsidian)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(vault_watcher_loop())
    yield
    task.cancel()


app = FastAPI(title="Product Brain", lifespan=lifespan)


@app.get("/")
def serve_home():
    index_path = BASE_DIR / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/state")
def get_state():
    repos = []
    for repo_dir in sorted(REPOS_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        out_dir = repo_dir / "graphify-out"
        # Skip incomplete clones where graph.json was never generated
        if not (out_dir / "graph.json").exists():
            continue
        report_file = out_dir / "GRAPH_REPORT.md"
        html_file = out_dir / "graph.html"
        repos.append({
            "name": repo_dir.name,
            "report": report_file.read_text(encoding="utf-8") if report_file.exists() else "No report generated yet.",
            "has_html": html_file.exists(),
        })

    vault_dir = get_vault_dir()
    pms = []
    pms_vault = vault_dir / "PMs"
    pm_names = set()
    for d in PMS_DIR.iterdir():
        if d.is_dir():
            pm_names.add(d.name)
    if pms_vault.exists():
        for d in pms_vault.iterdir():
            if d.is_dir():
                pm_names.add(d.name)

    for pm_name in sorted(pm_names):
        docs = set()
        for folder in (PMS_DIR / pm_name / "docs", pms_vault / pm_name):
            if folder.exists():
                for f in folder.glob("*"):
                    if f.is_file() and not f.name.startswith("00-") and f.name != "graph.canvas":
                        docs.add(f.name)
        pms.append({"name": pm_name, "docs": sorted(docs)})

    notes = []
    if vault_dir.exists():
        if (vault_dir / "00-Home.md").exists():
            notes.append({"rel_path": "00-Home.md", "category": "Overview", "title": "00-Home.md"})
        for r in repos:
            rep_rel = f"Repos/{r['name']}/00-Architecture-Report.md"
            if (vault_dir / rep_rel).exists():
                notes.append({"rel_path": rep_rel, "category": "Repo Architecture", "title": f"{r['name']} — Architecture"})
        prds_dir = vault_dir / "PRDs"
        if prds_dir.exists():
            for f in sorted(prds_dir.glob("*.md"), reverse=True):
                notes.append({"rel_path": f"PRDs/{f.name}", "category": "Drafted PRDs", "title": f.name})
        if pms_vault.exists():
            for pm_folder in sorted(pms_vault.iterdir()):
                if not pm_folder.is_dir():
                    continue
                for f in sorted(pm_folder.glob("*.md")):
                    if not f.name.startswith("00-"):
                        notes.append({
                            "rel_path": f"PMs/{pm_folder.name}/{f.name}",
                            "category": f"PM: {pm_folder.name}",
                            "title": f.name,
                        })

    total_nodes = 0
    global_graph_json = vault_dir / "_graphify" / "global_graph.json"
    if global_graph_json.exists():
        try:
            total_nodes = len(json.loads(global_graph_json.read_text(encoding="utf-8")).get("nodes", []))
        except Exception:
            pass

    return {
        "repos": repos,
        "pms": pms,
        "notes": notes,
        "vault_path": str(vault_dir),
        "claude_installed": bool(find_claude_binary()),
        "total_nodes": total_nodes,
    }


@app.get("/api/notes/content")
def get_note_content(rel_path: str):
    target = resolve_vault_file(rel_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Note not found in Obsidian vault")
    return {
        "rel_path": rel_path,
        "content": target.read_text(encoding="utf-8", errors="ignore"),
    }


@app.post("/api/notes/save")
def save_note_content(req: NoteSaveRequest):
    target = resolve_vault_file(req.rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    sync_all_to_obsidian()
    return {"status": "ok", "rel_path": req.rel_path}


@app.post("/api/vault")
def update_vault(req: VaultConfigRequest):
    if not req.vault_path.strip():
        raise HTTPException(status_code=400, detail="Vault path cannot be empty")
    try:
        set_vault_dir(req.vault_path)
        result = sync_all_to_obsidian()
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/obsidian/sync")
def trigger_obsidian_sync():
    try:
        result = sync_all_to_obsidian()
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/obsidian/open")
def open_in_obsidian(req: OpenNoteRequest):
    vault = get_vault_dir()
    vault.mkdir(parents=True, exist_ok=True)
    target = resolve_vault_file(req.rel_path) if req.rel_path else (vault / "00-Home.md")
    if not target.exists():
        target = vault
    uri = f"obsidian://open?path={urllib.parse.quote(str(target))}"
    subprocess.Popen(["open", uri])
    return {"status": "ok", "uri": uri}


@app.post("/api/repos")
def sync_repo(req: RepoRequest):
    url = req.repo_url.strip().rstrip("/")
    if not (url.startswith("https://") or url.startswith("http://") or url.startswith("git@")):
        raise HTTPException(status_code=400, detail="Please enter a valid GitHub HTTPS or SSH URL")

    repo_name = sanitize_name(url.split("/")[-1].replace(".git", ""))
    repo_dir = REPOS_DIR / repo_name

    clone_url = url
    if req.github_token and url.startswith("https://"):
        clone_url = url.replace("https://", f"https://{req.github_token}@", 1)

    try:
        if repo_dir.exists():
            run_cmd(["git", "pull"], cwd=repo_dir)
        else:
            run_cmd(["git", "clone", "--depth", "1", clone_url, str(repo_dir)])

        ignore_file = repo_dir / ".graphifyignore"
        if not ignore_file.exists():
            ignore_file.write_text(DEFAULT_GRAPHIFY_IGNORE, encoding="utf-8")

        run_cmd(["graphify", "extract", ".", "--code-only"], cwd=repo_dir)
        vault_info = sync_all_to_obsidian()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ok", "repo": repo_name, **vault_info}


@app.delete("/api/repos/{repo_name}")
def delete_repo(repo_name: str):
    safe = sanitize_name(repo_name)
    repo_dir = REPOS_DIR / safe
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    vault_repo = get_vault_dir() / "Repos" / safe
    if vault_repo.exists():
        shutil.rmtree(vault_repo)
    sync_all_to_obsidian()
    return {"status": "ok", "deleted": safe}


@app.post("/api/repos/refresh-all")
def refresh_all_repos():
    refreshed = []
    errors = []
    for repo_dir in sorted(REPOS_DIR.iterdir()):
        if not repo_dir.is_dir():
            continue
        try:
            run_cmd(["git", "pull"], cwd=repo_dir)
            run_cmd(["graphify", "extract", ".", "--code-only"], cwd=repo_dir)
            refreshed.append(repo_dir.name)
        except Exception as exc:
            errors.append(f"{repo_dir.name}: {exc}")

    vault_info = sync_all_to_obsidian()
    return {"status": "ok", "refreshed": refreshed, "errors": errors, **vault_info}


@app.post("/api/pms/{pm_name}/docs")
async def upload_pm_doc(pm_name: str, file: UploadFile = File(...)):
    safe_pm = sanitize_name(pm_name)
    if not safe_pm:
        raise HTTPException(status_code=400, detail="Invalid PM name")

    docs_dir = PMS_DIR / safe_pm / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / sanitize_name(file.filename or "prd.md")

    content = await file.read()
    dest.write_bytes(content)

    vault_info = sync_all_to_obsidian()
    return {"status": "ok", "pm": safe_pm, "file": dest.name, **vault_info}


@app.get("/repos/{repo_name}/graph.html")
def serve_repo_graph_html(repo_name: str):
    html_path = REPOS_DIR / sanitize_name(repo_name) / "graphify-out" / "graph.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Interactive graph HTML not found for this repo")
    return FileResponse(html_path)


@app.post("/api/claude")
def ask_claude_code(req: ClaudeAskRequest):
    vault = get_vault_dir()
    if not (vault / "CLAUDE.md").exists():
        sync_all_to_obsidian()
    settings_path = write_claude_settings(vault)

    # Build a scoped graph if specific repos/PMs are checked; otherwise use global_graph.json
    meta_dir = vault / "_graphify"
    scoped_graph = meta_dir / "scoped_query_graph.json"
    selected_paths: List[Path] = []

    for r in req.selected_repos:
        g = REPOS_DIR / sanitize_name(r) / "graphify-out" / "graph.json"
        if g.exists():
            selected_paths.append(g)
    for p in req.selected_pms:
        g = PMS_DIR / sanitize_name(p) / "graphify-out" / "graph.json"
        if g.exists():
            selected_paths.append(g)

    active_graph_rel = "_graphify/global_graph.json"
    if selected_paths:
        if len(selected_paths) == 1:
            shutil.copyfile(selected_paths[0], scoped_graph)
        else:
            run_cmd(["graphify", "merge-graphs", *[str(p) for p in selected_paths], "--out", str(scoped_graph)])
        active_graph_rel = "_graphify/scoped_query_graph.json"

    active_graph_abs = vault / active_graph_rel
    graph_preview = ""
    if active_graph_abs.exists():
        try:
            if req.mode == "path":
                parts = [p.strip() for p in re.split(r"->|,|\bto\b", req.prompt, maxsplit=1) if p.strip()]
                if len(parts) >= 2:
                    graph_preview = run_cmd(
                        ["graphify", "path", parts[0], parts[1], "--graph", str(active_graph_abs)],
                        allow_codes=(0, 3, 4),
                    )
            else:
                graph_preview = run_cmd(
                    ["graphify", "query", req.prompt, "--graph", str(active_graph_abs), "--budget", "2500"],
                    allow_codes=(0, 3, 4),
                )
        except Exception:
            pass

    mode_instructions = {
        "ask": "Answer the PM's question concisely using the knowledge graph and Obsidian notes. Tag claims with [Code Verified] or [Peer PRD].",
        "overlap": "Audit peer PM notes in `PMs/` and `PRDs/` plus code modules in `Repos/` for any overlapping features, prior decisions, or conflicts related to the user's topic.",
        "path": "Trace the dependency and conceptual path between the requested entities using `graphify path` and explain how they connect.",
        "prd": (
            "Draft a complete, production-ready PRD and save it directly as a new Markdown file in `PRDs/` "
            "(e.g., `PRDs/2026-11-24-<feature-slug>.md`) using the Write tool, with Obsidian `[[wikilinks]]` to the referenced services and peer PRDs. "
            "Include: 1) Problem & Goals, 2) Current System Architecture ([Code Verified]), 3) Prior Peer PM Context ([Peer PRD]), "
            "4) Functional Requirements, 5) Technical Risks & Edge Cases, and 6) Rollout Metrics. "
            "In your final reply, summarize the PRD and explicitly state the created file path `PRDs/<filename>.md`."
        ),
    }

    full_prompt = (
        f"Active Graph File: `{active_graph_rel}`\n"
        f"Selected Repos: {req.selected_repos or 'All'} | Selected PMs: {req.selected_pms or 'All'}\n"
        f"Task Mode: {req.mode.upper()} — {mode_instructions.get(req.mode, mode_instructions['ask'])}\n\n"
        f"PM Request: {req.prompt}\n\n"
        f"Pre-Retrieved Graphify Subgraph:\n{graph_preview}"
    )

    claude_bin = find_claude_binary()
    if not claude_bin:
        return {
            "answer": (
                "Claude Code CLI (`claude`) was not detected on your PATH, so here is the verified Graphify subgraph:\n\n"
                + (graph_preview or "No matching graph nodes found yet. Sync a repository or upload a PRD first.")
            ),
            "session_id": None,
            "created_prd": None,
            "fallback": True,
        }

    cmd = [
        claude_bin,
        "-p",
        full_prompt,
        "--output-format",
        "json",
        "--settings",
        str(settings_path),
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read,Grep,Glob,Write,Edit,Bash(graphify *)",
    ]
    if req.session_id:
        cmd.extend(["--resume", req.session_id])

    try:
        raw_out = run_cmd(cmd, cwd=vault, timeout=300)
        parsed = json.loads(raw_out.strip().splitlines()[-1])
        answer_text = parsed.get("result") or raw_out
        new_session_id = parsed.get("session_id") or req.session_id
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Claude Code error: {exc}")

    sync_all_to_obsidian()
    created_prd = None
    prd_match = re.search(r"(PRDs/[a-zA-Z0-9._-]+\.md)", answer_text)
    if prd_match and (vault / prd_match.group(1)).exists():
        created_prd = prd_match.group(1)

    return {
        "answer": answer_text,
        "session_id": new_session_id,
        "created_prd": created_prd,
        "fallback": False,
    }