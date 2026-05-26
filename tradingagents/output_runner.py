from __future__ import annotations

import argparse
import html
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional


DEFAULT_TICKER = "SPY"
DEFAULT_PUBLIC_HOST = os.getenv("TICKER_AGENTS_PUBLIC_HOST", "10.17.17.98")
DEFAULT_PORT = int(os.getenv("TICKER_AGENTS_OUTPUT_PORT", "8765"))
FINAL_BEGIN = "=== FINAL_DECISION_MARKDOWN_BEGIN ==="
FINAL_END = "=== FINAL_DECISION_MARKDOWN_END ==="
STACK_EXAMPLES = {
    "grounded": "run_grounded_stack.py",
    "fmp": "run_fmp_mcp_stack.py",
}


@dataclass
class RunPaths:
    run_dir: Path
    console_txt: Path
    metadata_json: Path
    live_html: Path
    index_html: Path
    final_md: Path
    final_html: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def resolve_run_request(
    ticker: Optional[str], analysis_date: Optional[str], *, today: Optional[date] = None
) -> tuple[str, str]:
    today = today or date.today()
    resolved_ticker = (ticker or DEFAULT_TICKER).strip().upper()
    resolved_date = (analysis_date or today.isoformat()).strip()
    datetime.strptime(resolved_date, "%Y-%m-%d")
    return resolved_ticker, resolved_date


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_run_slug(ticker: str, analysis_date: str, started_at: datetime) -> str:
    host = slugify(socket.gethostname())
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    return slugify(f"{ticker}-{analysis_date}-{stamp}-{host}-pid{os.getpid()}")


def build_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        run_dir=run_dir,
        console_txt=run_dir / "console.txt",
        metadata_json=run_dir / "metadata.json",
        live_html=run_dir / "live.html",
        index_html=run_dir / "index.html",
        final_md=run_dir / "final.md",
        final_html=run_dir / "final.html",
    )


def build_app_command(repo_root: Path, ticker: str, analysis_date: str, stack: str) -> list[str]:
    example_name = STACK_EXAMPLES[stack]
    return [
        str(repo_root / ".venv" / "bin" / "python"),
        str(repo_root / "examples" / example_name),
        ticker,
        analysis_date,
    ]


def extract_final_decision(log_text: str) -> Optional[str]:
    start = log_text.rfind(FINAL_BEGIN)
    end = log_text.rfind(FINAL_END)
    if start == -1 or end == -1 or end <= start:
        return None
    body = log_text[start + len(FINAL_BEGIN):end].strip()
    return body or None


def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_live_html(title: str) -> str:
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"refresh\" content=\"15\" />
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{ background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; margin:0; }}
    header {{ padding:20px 24px; border-bottom:1px solid #223; position:sticky; top:0; background:#0b1020; z-index:10; }}
    h1 {{ margin:0 0 8px 0; font-size:20px; }}
    .meta {{ color:#9ca3af; font-size:14px; display:flex; gap:16px; flex-wrap:wrap; }}
    main {{ padding:24px; display:grid; gap:18px; max-width:1200px; margin:0 auto; }}
    a {{ color:#93c5fd; }}
    .pill {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#1f2937; color:#d1d5db; font-size:12px; }}
    .grid {{ display:grid; grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr); gap:18px; align-items:start; }}
    .panel {{ background:#111827; border:1px solid #1f2937; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.18); }}
    .panel h2 {{ margin:0; font-size:16px; padding:16px 18px; border-bottom:1px solid #1f2937; }}
    .panel-body {{ padding:16px 18px; }}
    .stats {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }}
    .stat {{ background:#0b1222; border:1px solid #243045; border-radius:12px; padding:12px; }}
    .stat-label {{ display:block; font-size:12px; color:#94a3b8; margin-bottom:6px; }}
    .stat-value {{ font-size:22px; font-weight:700; }}
    .feed {{ display:grid; gap:12px; }}
    .entry {{ border:1px solid #22304a; border-left-width:6px; border-radius:12px; background:#0d1528; overflow:hidden; }}
    .entry-header {{ padding:10px 14px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#cbd5e1; background:rgba(255,255,255,.02); border-bottom:1px solid rgba(255,255,255,.05); }}
    .entry-body {{ padding:14px; white-space:pre-wrap; word-break:break-word; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; line-height:1.55; }}
    .entry-human {{ border-left-color:#38bdf8; background:#081824; }}
    .entry-ai {{ border-left-color:#818cf8; background:#0f1530; }}
    .entry-tool {{ border-left-color:#f59e0b; background:#201506; }}
    .entry-final {{ border-left-color:#22c55e; background:#071b12; }}
    .entry-final .entry-body {{ font-family:inherit; font-size:14px; }}
    .entry-system {{ border-left-color:#64748b; background:#121826; }}
    .markdown {{ line-height:1.65; }}
    .markdown h1, .markdown h2, .markdown h3 {{ line-height:1.25; }}
    .markdown code, .markdown pre {{ background:#0b1020; border-radius:8px; }}
    .markdown pre {{ padding:16px; overflow:auto; }}
    .markdown table {{ border-collapse:collapse; width:100%; }}
    .markdown th, .markdown td {{ border:1px solid #263244; padding:8px 10px; text-align:left; vertical-align:top; }}
    .markdown blockquote {{ border-left:4px solid #374151; margin-left:0; padding-left:16px; color:#cbd5e1; }}
    .raw-log {{ max-height:70vh; overflow:auto; white-space:pre-wrap; word-break:break-word; background:#0b1222; border:1px solid #243045; border-radius:12px; padding:14px; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; line-height:1.5; }}
    .muted {{ color:#94a3b8; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class=\"meta\">
      <span id=\"status\" class=\"pill\">Loading…</span>
      <span id=\"duration\">Duration: --</span>
      <a href=\"console.txt\">Raw text</a>
      <a href=\"index.html\">Run index</a>
      <a href=\"final.html\">Rendered final decision</a>
    </div>
  </header>
  <main>
    <div class=\"grid\">
      <section class=\"panel\">
        <h2>Live structured view</h2>
        <div class=\"panel-body\">
          <div class=\"feed\" id=\"feed\">
            <div class=\"entry entry-system\"><div class=\"entry-header\">Loading</div><div class=\"entry-body\">Waiting for output…</div></div>
          </div>
        </div>
      </section>
      <aside class=\"panel\">
        <h2>Run summary</h2>
        <div class=\"panel-body\">
          <div class=\"stats\">
            <div class=\"stat\"><span class=\"stat-label\">Human messages</span><span class=\"stat-value\" id=\"humanCount\">0</span></div>
            <div class=\"stat\"><span class=\"stat-label\">Tool calls</span><span class=\"stat-value\" id=\"toolCount\">0</span></div>
            <div class=\"stat\"><span class=\"stat-label\">AI blocks</span><span class=\"stat-value\" id=\"aiCount\">0</span></div>
            <div class=\"stat\"><span class=\"stat-label\">Final sections</span><span class=\"stat-value\" id=\"finalCount\">0</span></div>
          </div>
          <p class=\"muted\" style=\"margin-top:16px\">Color coding: blue = human, amber = tool calls, indigo = AI output, green = final recommendation.</p>
          <details style=\"margin-top:18px\">
            <summary>Raw log</summary>
            <div id=\"log\" class=\"raw-log\">Loading output…</div>
          </details>
        </div>
      </aside>
    </div>
  </main>
  <script>
    const logEl = document.getElementById('log');
    const feedEl = document.getElementById('feed');
    const statusEl = document.getElementById('status');
    const durationEl = document.getElementById('duration');
    const humanCountEl = document.getElementById('humanCount');
    const toolCountEl = document.getElementById('toolCount');
    const aiCountEl = document.getElementById('aiCount');
    const finalCountEl = document.getElementById('finalCount');

    function escapeHtml(value) {{
      return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }}

    function renderInlineMarkdown(md) {{
      let body = escapeHtml(md.trim());
      body = body.replace(/^###\s+(.*)$/gm, '<h3>$1</h3>');
      body = body.replace(/^##\s+(.*)$/gm, '<h2>$1</h2>');
      body = body.replace(/^#\s+(.*)$/gm, '<h1>$1</h1>');
      body = body.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      body = body.replace(/\*(.+?)\*/g, '<em>$1</em>');
      body = body.replace(/`([^`]+)`/g, '<code>$1</code>');
      body = body.replace(/\n\n/g, '</p><p>');
      body = body.replace(/\n-\s+/g, '\n• ');
      return '<div class="markdown"><p>' + body + '</p></div>';
    }}

    function buildEntry(kind, label, body, isMarkdown=false) {{
      const wrapper = document.createElement('div');
      wrapper.className = 'entry entry-' + kind;
      const header = document.createElement('div');
      header.className = 'entry-header';
      header.textContent = label;
      const content = document.createElement('div');
      content.className = 'entry-body';
      if (isMarkdown) {{
        content.innerHTML = renderInlineMarkdown(body);
      }} else {{
        content.textContent = body.trim() || '…';
      }}
      wrapper.appendChild(header);
      wrapper.appendChild(content);
      return wrapper;
    }}

    function parseStructuredEntries(logText) {{
      const entries = [];
      const finalMatches = [...logText.matchAll(/=== FINAL_DECISION_MARKDOWN_BEGIN ===([\s\S]*?)=== FINAL_DECISION_MARKDOWN_END ===/g)];
      const cleaned = logText.replace(/=== FINAL_DECISION_MARKDOWN_BEGIN ===[\s\S]*?=== FINAL_DECISION_MARKDOWN_END ===/g, '').trim();
      const humanHeader = '================================ Human Message =================================';
      const aiHeader = '================================== Ai Message ==================================';
      const lines = cleaned.split('\n');
      let current = [];
      let currentKind = 'system';
      let currentLabel = 'System / setup';

      function flushCurrent() {{
        const body = current.join('\n').trim();
        if (body) entries.push({{ kind: currentKind, label: currentLabel, body }});
        current = [];
      }}

      for (const line of lines) {{
        if (line === humanHeader) {{
          flushCurrent();
          currentKind = 'human';
          currentLabel = 'Human message';
          continue;
        }}
        if (line === aiHeader) {{
          flushCurrent();
          currentKind = 'ai';
          currentLabel = 'AI output';
          continue;
        }}
        current.push(line);
      }}
      flushCurrent();

      const expanded = [];
      for (const entry of entries) {{
        if (entry.kind !== 'ai') {{
          expanded.push(entry);
          continue;
        }}
        const toolRegex = /<tool_call>[\s\S]*?<\/tool_call>/g;
        const toolMatches = [...entry.body.matchAll(toolRegex)];
        const plain = entry.body.replace(toolRegex, '').trim();
        if (plain) expanded.push({{ kind:'ai', label:'AI output', body: plain }});
        for (const match of toolMatches) {{
          expanded.push({{ kind:'tool', label:'Tool call', body: match[0] }});
        }}
      }}

      for (const match of finalMatches) {{
        expanded.push({{ kind:'final', label:'Final recommendation', body: match[1].trim(), markdown: true }});
      }}
      return expanded;
    }}

    function renderEntries(logText) {{
      const entries = parseStructuredEntries(logText);
      feedEl.replaceChildren();
      if (!entries.length) {{
        feedEl.appendChild(buildEntry('system', 'No output yet', 'Waiting for output…'));
      }}
      let human = 0, tool = 0, ai = 0, final = 0;
      for (const entry of entries) {{
        if (entry.kind === 'human') human += 1;
        if (entry.kind === 'tool') tool += 1;
        if (entry.kind === 'ai') ai += 1;
        if (entry.kind === 'final') final += 1;
        feedEl.appendChild(buildEntry(entry.kind, entry.label, entry.body, Boolean(entry.markdown)));
      }}
      humanCountEl.textContent = String(human);
      toolCountEl.textContent = String(tool);
      aiCountEl.textContent = String(ai);
      finalCountEl.textContent = String(final);
    }}

    async function refresh() {{
      try {{
        const [logResp, metaResp] = await Promise.all([
          fetch('console.txt?ts=' + Date.now()),
          fetch('metadata.json?ts=' + Date.now()),
        ]);
        const logText = await logResp.text();
        logEl.textContent = logText;
        renderEntries(logText);
        const meta = await metaResp.json();
        statusEl.textContent = meta.status || 'unknown';
        durationEl.textContent = 'Duration: ' + (meta.duration_hms || '--');
        window.scrollTo(0, document.body.scrollHeight);
      }} catch (err) {{
        statusEl.textContent = 'refresh failed';
      }}
    }}
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def render_index_html(metadata: dict) -> str:
    title = html.escape(metadata.get("title", "Ticker Agents Run"))
    links = [
        ("Live HTML", "live.html"),
        ("Raw console text", "console.txt"),
        ("Run metadata", "metadata.json"),
    ]
    if metadata.get("has_final_markdown"):
        links.append(("Rendered final decision", "final.html"))
        links.append(("Final decision markdown", "final.md"))
    link_html = "".join(f'<li><a href="{href}">{label}</a></li>' for label, href in links)
    details = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in [
            ("ticker", metadata.get("ticker")),
            ("analysis_date", metadata.get("analysis_date")),
            ("status", metadata.get("status")),
            ("started_at", metadata.get("started_at")),
            ("finished_at", metadata.get("finished_at") or "—"),
            ("duration", metadata.get("duration_hms") or "—"),
            ("exit_code", metadata.get("exit_code")),
            ("command", metadata.get("command")),
        ]
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; padding:24px; }}
    a {{ color:#93c5fd; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
    th, td {{ border: 1px solid #1f2937; padding: 8px 12px; text-align: left; vertical-align: top; }}
    th {{ width: 180px; background:#111827; }}
    .box {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:20px; max-width:1100px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class=\"box\">
    <p>This run stores both CLI-friendly text and browser-friendly HTML output.</p>
    <ul>{link_html}</ul>
    <table>{details}</table>
  </div>
</body>
</html>
"""


def render_markdown_html(title: str, body: str) -> str:
    markdown_warning = ""
    try:
        import markdown

        rendered = markdown.markdown(body, extensions=["fenced_code", "tables", "sane_lists"])
    except ModuleNotFoundError:
        markdown_warning = (
            "<p><strong>Note:</strong> Python-Markdown was unavailable, so this final output is shown "
            "in a plain-text fallback view.</p>"
        )
        rendered = f"<pre>{html.escape(body)}</pre>"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; }}
    article {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; line-height: 1.65; }}
    h1,h2,h3 {{ line-height:1.25; }}
    code, pre {{ background:#111827; border-radius:8px; }}
    pre {{ padding:16px; overflow:auto; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ border:1px solid #1f2937; padding:8px 10px; text-align:left; vertical-align:top; }}
    a {{ color:#93c5fd; }}
    blockquote {{ border-left:4px solid #374151; margin-left:0; padding-left:16px; color:#cbd5e1; }}
  </style>
</head>
<body>
  <article>
    <p><a href=\"index.html\">← Back to run index</a></p>
    {markdown_warning}
    {rendered}
  </article>
</body>
</html>
"""


def ensure_http_server(output_root: Path, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            return
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", str(output_root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.5)


def stream_command(command: list[str], cwd: Path, env: dict[str, str], out_handle, capture: list[str]) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_handle.write(line)
        out_handle.flush()
        capture.append(line)
    return process.wait()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ticker agents with captured output.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, defaults to SPY")
    parser.add_argument("analysis_date", nargs="?", help="Analysis date YYYY-MM-DD, defaults to today")
    parser.add_argument(
        "--stack",
        choices=sorted(STACK_EXAMPLES),
        default=os.getenv("TICKER_AGENTS_STACK", "fmp"),
        help="Example stack to run: fmp (default, Massive+FMP hybrid) or grounded",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port for the output viewer")
    return parser.parse_args(argv)


def run(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ticker, analysis_date = resolve_run_request(args.ticker, args.analysis_date)

    def _handle_signal(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "output"
    output_root.mkdir(exist_ok=True)

    started_at = datetime.now()
    slug = build_run_slug(ticker, analysis_date, started_at)
    paths = build_paths(output_root / slug)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    ensure_http_server(output_root, args.port)

    live_url = f"http://{DEFAULT_PUBLIC_HOST}:{args.port}/{slug}/live.html"
    index_url = f"http://{DEFAULT_PUBLIC_HOST}:{args.port}/{slug}/index.html"
    raw_url = f"http://{DEFAULT_PUBLIC_HOST}:{args.port}/{slug}/console.txt"

    metadata = {
        "title": f"Ticker Agents Run: {ticker} @ {analysis_date}",
        "ticker": ticker,
        "analysis_date": analysis_date,
        "stack": args.stack,
        "status": "running",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": None,
        "duration_hms": "00:00:00",
        "exit_code": None,
        "slug": slug,
        "command": None,
        "live_url": live_url,
        "index_url": index_url,
        "raw_url": raw_url,
        "has_final_markdown": False,
        "artifact_warning": None,
    }
    write_metadata(paths.metadata_json, metadata)
    paths.live_html.write_text(render_live_html(metadata["title"]), encoding="utf-8")
    paths.index_html.write_text(render_index_html(metadata), encoding="utf-8")

    banner = (
        f"Run directory: {paths.run_dir}\n"
        f"Live HTML: {live_url}\n"
        f"Index: {index_url}\n"
        f"Raw text: {raw_url}\n\n"
    )
    print(banner, end="")

    combined_output: list[str] = []
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")
    try:
        with paths.console_txt.open("w", encoding="utf-8") as out_handle:
            header = (
                f"# Ticker Agents Run\n"
                f"ticker: {ticker}\n"
                f"analysis_date: {analysis_date}\n"
                f"started_at: {metadata['started_at']}\n"
                f"live_html: {live_url}\n"
                f"raw_text: {raw_url}\n\n"
            )
            out_handle.write(banner)
            out_handle.write(header)
            out_handle.flush()
            combined_output.extend([banner, header])

            sync_exit = stream_command(["uv", "sync"], repo_root, env, out_handle, combined_output)
            if sync_exit != 0:
                metadata["status"] = "failed"
                metadata["exit_code"] = sync_exit
            else:
                app_command = build_app_command(repo_root, ticker, analysis_date, args.stack)
                metadata["command"] = " ".join(app_command)
                write_metadata(paths.metadata_json, metadata)
                app_exit = stream_command(app_command, repo_root, env, out_handle, combined_output)
                metadata["status"] = "completed" if app_exit == 0 else "failed"
                metadata["exit_code"] = app_exit
    except KeyboardInterrupt as exc:
        metadata["status"] = "interrupted"
        metadata["exit_code"] = 130
        interruption_note = f"\n\n# Interrupted\n{exc}\n"
        combined_output.append(interruption_note)
        with paths.console_txt.open("a", encoding="utf-8") as out_handle:
            out_handle.write(interruption_note)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    # Recompute elapsed correctly outside file handle to include whole run.
    finished_at = datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()
    metadata["finished_at"] = finished_at.isoformat(timespec="seconds")
    metadata["duration_hms"] = format_duration(duration_seconds)

    full_text = "".join(combined_output)
    final_md = extract_final_decision(full_text)
    try:
        if final_md:
            paths.final_md.write_text(final_md + "\n", encoding="utf-8")
            paths.final_html.write_text(
                render_markdown_html(f"Final Decision: {ticker} @ {analysis_date}", final_md),
                encoding="utf-8",
            )
            metadata["has_final_markdown"] = True
        else:
            metadata["has_final_markdown"] = False
    except Exception as exc:
        metadata["has_final_markdown"] = paths.final_md.exists()
        metadata["artifact_warning"] = f"final artifact rendering failed: {exc}"
        if metadata["status"] == "completed":
            metadata["status"] = "completed_with_warnings"
    finally:
        with paths.console_txt.open("a", encoding="utf-8") as out_handle:
            out_handle.write(f"\n\n# Run summary\n")
            out_handle.write(f"status: {metadata['status']}\n")
            out_handle.write(f"finished_at: {metadata['finished_at']}\n")
            out_handle.write(f"duration: {metadata['duration_hms']}\n")
            out_handle.write(f"exit_code: {metadata['exit_code']}\n")
            if metadata.get("artifact_warning"):
                out_handle.write(f"artifact_warning: {metadata['artifact_warning']}\n")

        write_metadata(paths.metadata_json, metadata)
        paths.index_html.write_text(render_index_html(metadata), encoding="utf-8")

    print()
    print(f"Completed with status={metadata['status']} in {metadata['duration_hms']}")
    print(f"Live HTML: {live_url}")
    print(f"Index: {index_url}")
    print(f"Raw text: {raw_url}")
    if metadata["has_final_markdown"]:
        print(f"Rendered final decision: http://{DEFAULT_PUBLIC_HOST}:{args.port}/{slug}/final.html")
    return int(metadata["exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(run())
