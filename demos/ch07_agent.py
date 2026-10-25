"""
The agent loop of Chapter 7 (Agentic RAG): tools, tool calling, and record/replay.

    tools:   wikipedia_search  hybrid retrieval over the nobel-2025 knowledge base (ch07-02)
             people_search     a small people directory with structured JSON records
             web_search        the live Wikipedia search API, standing in for a web search

    runtime: Runtime(run).llm(messages, tools)   one tool-calling turn with Claude
             Runtime(run).call_tool(name, args)  run a tool and return its result as text
             Runtime(run).ask(prompt)            a plain completion (guardrail checks)

A live run calls Bedrock and the tools. Without AWS access, the runtime replays a run
recorded earlier from data/ch07/agent_runs.json: the same model replies and the same
tool results, step by step. Set MMIR_RECORD=1 to record live runs into that file.

Usage:
    from ch07_agent import Runtime, make_tools, tool_result
    tools = make_tools(search, parents)
    rt = Runtime("nobel", tools)
    reply = rt.llm(messages, TOOLS, system=SYSTEM)
"""

import html
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from shared.llm import invoke_claude, invoke_claude_tools, AWSConnectionError

RUNS_FILE = Path(__file__).parent / "data" / "ch07" / "agent_runs.json"


# ─── Tools ───────────────────────────────────────────────────────────────────

# A people directory as an HR or CRM system would hold it: one structured record per person.
# The entries follow the pinned Wikipedia revisions of the knowledge base. It is deliberately
# incomplete: John M. Martinis has no record, so the agent must find him another way.
PEOPLE = [
    {"name": "John Clarke", "born": 1942, "nationality": "United Kingdom",
     "affiliation": "University of California, Berkeley", "position": "Professor Emeritus",
     "field": "superconducting electronics"},
    {"name": "Michel Devoret", "born": 1953, "nationality": "France, United States",
     "affiliation": "University of California, Santa Barbara", "position": "Professor of Physics",
     "other_affiliations": ["Yale University (Professor Emeritus of Applied Physics)",
                            "Google Quantum AI (Chief Scientist for Quantum Hardware)"],
     "field": "superconducting quantum circuits"},
    {"name": "Geoffrey Hinton", "born": 1947, "nationality": "United Kingdom, Canada",
     "affiliation": "University of Toronto", "position": "University Professor Emeritus",
     "field": "artificial neural networks"},
    {"name": "John Hopfield", "born": 1933, "nationality": "United States",
     "affiliation": "Princeton University", "position": "Professor Emeritus",
     "field": "associative neural networks"},
    {"name": "Francis Halzen", "born": 1944, "nationality": "Belgium, United States",
     "affiliation": "University of Wisconsin-Madison", "position": "Professor of Physics",
     "field": "neutrino astronomy (IceCube)"},
]


def _norm(name: str) -> set[str]:
    """Lower-case name tokens: 'John M. Martinis' -> {'john', 'm', 'martinis'}."""
    return set(re.findall(r"[a-z]+", name.lower()))


def make_tools(search, parents: list[dict]) -> dict:
    """
    The three tool implementations, keyed by tool name. Each takes the tool's arguments
    and returns a JSON-serializable result.

    `search(query, k)` is the hybrid retrieval of ch07-02 returning parent ids.
    """

    def wikipedia_search(query: str) -> list[dict]:
        return [{"source": f"{parents[p]['title']} > {parents[p]['section']}",
                 "url": parents[p]["url"], "text": parents[p]["body"]}
                for p in search(query, 3)]

    def people_search(name: str) -> dict:
        wanted = _norm(name)
        for person in PEOPLE:                       # match on the surname
            if person["name"].split()[-1].lower() in wanted:
                return person
        return {"error": f"no record for {name!r} in the people directory"}

    def web_search(query: str) -> list[dict] | dict:
        import requests
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query,
                        "srlimit": 3, "format": "json"},
                headers={"User-Agent": "mmir-unibasel-course-demo/1.0"}, timeout=10)
            hits = r.json()["query"]["search"]
        except Exception as e:                          # offline, blocked, rate-limited
            return {"error": f"web search unavailable ({type(e).__name__})"}
        strip = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s))
        return [{"title": h["title"], "snippet": strip(h["snippet"]),
                 "url": f"https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')}"}
                for h in hits]

    return {"wikipedia_search": wikipedia_search, "people_search": people_search,
            "web_search": web_search}


def tool_result(call_id: str, result: str) -> dict:
    """The observation block that returns a tool's result to the model."""
    return {"type": "tool_result", "tool_use_id": call_id, "content": result}


# ─── Runtime: live calls with record/replay ──────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    """One model turn: its text, the tool calls it requests, and the raw message."""
    text: str
    tool_calls: list[ToolCall]
    message: dict
    usage: dict = field(default_factory=dict)


def load_runs(path: Path = RUNS_FILE) -> dict:
    """All recorded runs of a recording file, keyed by run name."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_runs(runs: dict, path: Path = RUNS_FILE):
    """Write recorded runs (only with MMIR_RECORD=1)."""
    if os.environ.get("MMIR_RECORD") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")


class Runtime:
    """
    Model and tool calls for one named run.

    The first model call decides the mode: live if Bedrock answers, replay if it does not.
    In replay mode every model reply and every tool result comes from the recording, so
    the run unfolds exactly as it did when it was recorded, including its measured time.

    `store` is an optional dict of recorded runs shared by many runtimes (see load_runs);
    without it, each runtime reads and writes RUNS_FILE itself.
    """

    def __init__(self, run: str, tools: dict, store: dict | None = None):
        self.run, self.tools, self.store = run, tools, store
        self.live = None
        self.events: list = []                       # recording (live) or playback (replay)
        self.trace: list[dict] = []                  # one row per tool call, for display
        self.calls = 0                               # model calls in this run
        self.tokens = 0                              # input + output tokens of llm calls
        self.seconds = 0.0                           # time spent in model and tool calls

    def _record(self, kind: str, fn):
        t0 = time.perf_counter()
        out = fn()
        seconds = time.perf_counter() - t0
        self.seconds += seconds
        self.events.append({"kind": kind, "value": out, "seconds": round(seconds, 3)})
        return out

    def _replay(self, kind: str):
        event = self.events.pop(0)
        assert event["kind"] == kind, f"recording out of step: expected {kind}, got {event['kind']}"
        self.seconds += event.get("seconds", 0.0)
        return event["value"]

    # model calls
    def _model(self, kind: str, fn):
        self.calls += 1
        if self.live is None:
            try:
                out = self._record(kind, fn)
                self.live = True
                return out
            except AWSConnectionError:
                self.live = False
                runs = self.store if self.store is not None else load_runs()
                self.events = list(runs.get(self.run, []))
                if not self.events:
                    raise RuntimeError(f"No AWS access and no recording of run {self.run!r}.")
        return self._record(kind, fn) if self.live else self._replay(kind)

    def llm(self, messages: list[dict], tools: list[dict], system: str = None) -> Reply:
        """One reason step: the model answers or requests tool calls."""
        raw = self._model("llm", lambda: invoke_claude_tools(messages, tools, system=system))
        usage = raw.get("usage", {})
        self.tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        blocks = [b for b in raw["content"] if b["type"] in ("text", "tool_use")]
        text = "".join(b["text"] for b in blocks if b["type"] == "text").strip()
        calls = [ToolCall(b["id"], b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"]
        return Reply(text, calls, {"role": "assistant", "content": blocks}, usage)

    def ask(self, prompt: str, system: str = None, max_tokens: int = 600) -> str:
        """A plain completion, used by the guardrail checks and judges."""
        return self._model("ask", lambda: invoke_claude(prompt, system=system,
                                                        max_tokens=max_tokens))

    # tool calls
    def call_tool(self, name: str, arguments: dict, reason: str = "") -> str:
        """One act step: run the tool and return its result as JSON text."""
        if self.live:
            result = self._record("tool", lambda: json.dumps(self.tools[name](**arguments),
                                                             ensure_ascii=False))
        else:
            result = self._replay("tool")
        self.trace.append({"step": self.calls, "reason": reason, "tool": name,
                           "arguments": arguments, "result": result})
        return result

    def save(self):
        """Store a live run (only with MMIR_RECORD=1): in `store`, or in RUNS_FILE."""
        if not self.live or os.environ.get("MMIR_RECORD") != "1":
            return
        if self.store is not None:
            self.store[self.run] = self.events
        else:
            runs = load_runs()
            runs[self.run] = self.events
            save_runs(runs)


# ─── The agent of ch07-04, for reuse ─────────────────────────────────────────

TOOLS = [
    {"name": "wikipedia_search",
     "description": "Search a Wikipedia knowledge base about the Nobel Prizes and return "
                    "the text of the 3 best matching passages with their source.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
    {"name": "people_search",
     "description": "Look up a person in the people directory. Returns JSON with name, "
                    "affiliation, and position, or an error if there is no record.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "web_search",
     "description": "Search the web. Returns titles and short snippets. "
                    "Use only when the other tools return nothing.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
]

SYSTEM = (
    "You answer questions about the Nobel Prizes with the tools provided. Base every "
    "statement on tool results, and name the source of each fact in brackets, like "
    "[John Clarke (physicist) > Introduction] or [people_search]. If the tools do not give "
    "the answer, say so."
)


def run_agent(question: str, rt: Runtime, max_steps: int = 8) -> str:
    """The reason-act-observe loop of ch07-04 on a given runtime; returns the answer."""
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        reply = rt.llm(messages, TOOLS, system=SYSTEM)
        messages.append(reply.message)
        if not reply.tool_calls:
            return reply.text
        results = [tool_result(call.id, rt.call_tool(call.name, call.arguments,
                                                     reply.text if i == 0 else ""))
                   for i, call in enumerate(reply.tool_calls)]
        messages.append({"role": "user", "content": results})
    return "I could not complete the answer within the step budget."
