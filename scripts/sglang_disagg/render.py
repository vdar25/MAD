#!/usr/bin/env python3
"""Render SGLang disaggregated P/D commands and environment from models.yaml.

Run settings are read from the environment as settings.sh leaves it (defaults
filled in, older names mapped); --set KEY=VALUE overrides one. They include
MODEL_NAME, xP, yD, PARALLEL_MODE (tp|wide_ep), KV_TRANSFER_BACKEND, IPADDRS
(comma-separated, rank order), MODEL_PATH, GPUS_PER_NODE, PREFILL_TP_SIZE,
DECODE_TP_SIZE, the ports, KV_IB_DEVICES, BENCHMARK_SCRIPT, ENABLE_RADIX_CACHE,
ENABLE_SERVER_METRICS, WORKER_STARTUP_TIMEOUT and the EXTRA_*_ARGS.

Commands:
  models          catalog models and the modes each supports
  validate        model, mode, backend and topology; non-zero exit on error
  plan [--json]   every node's role, environment and command (a dry run; needs no GPUs)
  node --rank R   shell assignments describing node R
  env --rank R    `export` lines for node R's catalog environment (existing env wins)
  argv --rank R   NUL-separated sglang.launch_server arguments for node R
  router-argv     NUL-separated sglang_router.launch_router arguments
  check           every flag and SGLANG_* variable the catalog can render exists
                  in this image's SGLang
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
MODES = ("tp", "wide_ep")
ROLES = ("prefill", "decode")

# Emitted by the launcher per node; a catalog entry setting one is a mistake.
LAUNCHER_FLAGS = frozenset({
    "--model-path", "--disaggregation-mode", "--host", "--port", "--tp-size",
    "--dp-size", "--ep-size", "--nnodes", "--node-rank", "--dist-init-addr",
    "--disaggregation-ib-device",
})

_RANGE = re.compile(r"\s*(\d+)\s*\.\.\s*(\d+)\s*")

Flags = Dict[str, object]
Env = Dict[str, Optional[str]]


class RenderError(Exception):
    pass


# --------------------------------------------------------------------------- catalog

def _flag_map(block, where: str) -> Flags:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise RenderError(f"{where}: expected a map of --flag: value, got {type(block).__name__}")
    for key in block:
        if not str(key).startswith("--"):
            raise RenderError(f"{where}: {key!r} is not a --flag")
        if key in LAUNCHER_FLAGS:
            raise RenderError(f"{where}: {key} is set by the launcher, not the catalog")
    return dict(block)


def _env_map(block, where: str) -> Env:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise RenderError(f"{where}: expected a map of NAME: value")
    return {str(k): (None if v is None else str(v)) for k, v in block.items()}


def _split(block, where: str) -> Tuple[Flags, Env]:
    """A block's --flags and its `env` map."""
    block = dict(block or {})
    env = _env_map(block.pop("env", None), f"{where}.env")
    return _flag_map(block, where), env


def _resolve_layer(cfg: Mapping, mode: str, role: str, where: str) -> Tuple[Flags, Env]:
    """base/env -> modes[mode] -> roles[role] -> roles[role][mode], later keys winning."""
    flags = _flag_map(cfg.get("base"), f"{where}.base")
    env = _env_map(cfg.get("env"), f"{where}.env")

    def add(block, at):
        f, e = _split(block, at)
        flags.update(f)
        env.update(e)

    modes = cfg.get("modes") or {}
    for name in modes:
        if name not in MODES:
            raise RenderError(f"{where}.modes: unknown mode {name!r} (expected one of {MODES})")
    add(modes.get(mode), f"{where}.modes.{mode}")

    roles = cfg.get("roles") or {}
    for name in roles:
        if name not in ROLES:
            raise RenderError(f"{where}.roles: unknown role {name!r} (expected one of {ROLES})")
    block = dict(roles.get(role) or {})
    by_mode = {m: block.pop(m) for m in MODES if m in block}
    for key in block:
        if key != "env" and not str(key).startswith("-"):
            raise RenderError(f"{where}.roles.{role}: {key!r} is neither a --flag, a mode nor env")
    add(block, f"{where}.roles.{role}")
    add(by_mode.get(mode), f"{where}.roles.{role}.{mode}")
    return flags, env


def _apply(acc: Dict, layer: Mapping, removes=(None, False)) -> None:
    for key, value in layer.items():
        if any(value is r for r in removes):
            acc.pop(key, None)
        else:
            acc[key] = value


def _inherit(parent: Mapping, child: Mapping) -> Dict:
    """Deep-merge `child` over `parent`; a null in the child drops the inherited key."""
    out = dict(parent)
    for key, value in child.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _inherit(out[key], value)
        else:
            out[key] = value
    return out


class Catalog:
    def __init__(self, models_path: Path):
        data = yaml.safe_load(models_path.read_text()) or {}
        self.defaults = data.get("defaults") or {}
        self.backends = data.get("backends") or {}
        self.overlays = data.get("overlays") or {}
        self._models = data.get("models") or {}

    def names(self) -> List[str]:
        return list(self._models)

    def model(self, name: str, _chain: Tuple[str, ...] = ()) -> Dict:
        if name not in self._models:
            raise RenderError(
                f"unknown MODEL_NAME {name!r}; models.yaml has: {', '.join(self._models)}")
        if name in _chain:
            raise RenderError(f"models.yaml: extends cycle {' -> '.join(_chain + (name,))}")
        cfg = dict(self._models[name] or {})
        parent = cfg.pop("extends", None)
        if parent:
            cfg = _inherit(self.model(parent, _chain + (name,)), cfg)
        return cfg

    def modes_for(self, name: str) -> List[str]:
        declared = self.model(name).get("modes") or {}
        return [m for m in MODES if m == "tp" or m in declared]

    def backend(self, name: str) -> Mapping:
        if name not in self.backends:
            raise RenderError(
                f"unknown KV_TRANSFER_BACKEND {name!r}; models.yaml backends: {', '.join(self.backends)}")
        return self.backends[name] or {}

    def _layers(self, name: str, mode: str, role: str) -> List[Tuple[Flags, Env]]:
        return [_resolve_layer(self.defaults, mode, role, "defaults"),
                _resolve_layer(self.model(name), mode, role, f"models.{name}")]

    def server_flags(self, name: str, mode: str, role: str, backend: str,
                     overlays: List[str] = ()) -> Flags:
        acc: Flags = {}
        for flags, _ in self._layers(name, mode, role):
            _apply(acc, flags)
        _apply(acc, _flag_map(self.backend(backend).get("args"), f"backends.{backend}.args"))
        for overlay in overlays:
            if overlay not in self.overlays:
                raise RenderError(f"models.yaml has no overlay {overlay!r}")
            _apply(acc, _flag_map(self.overlays[overlay], f"overlays.{overlay}"))
        return acc

    def server_env(self, name: str, mode: str, role: str) -> Dict[str, str]:
        acc: Dict[str, str] = {}
        for _, env in self._layers(name, mode, role):
            _apply(acc, env, removes=(None,))
        return acc


def _values(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in _values(item)]
    if isinstance(value, str):
        match = _RANGE.fullmatch(value)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            return [str(i) for i in range(lo, hi + 1)]
        return [value]
    return [str(value)]


def flags_to_argv(flags: Mapping[str, object]) -> List[str]:
    argv: List[str] = []
    for key, value in flags.items():
        if value is True:
            argv.append(key)
        elif value is not None and value is not False:
            argv.append(key)
            argv.extend(_values(value))
    return argv


# --------------------------------------------------------------------------- settings

def run_settings(env: Mapping[str, str]) -> Dict[str, str]:
    """`env` as settings.sh leaves it: defaults filled in, older names mapped."""
    result = subprocess.run(
        ["bash", "-c", 'source "$0" && env -0', str(HERE / "settings.sh")],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env}, capture_output=True)
    if result.returncode:
        raise RenderError(f"settings.sh failed: {result.stderr.decode().strip()}")
    pairs = (item.partition("=") for item in result.stdout.decode().split("\0"))
    return {key: value for key, sep, value in pairs if sep}


def _int(env: Mapping[str, str], key: str, required: bool = True) -> Optional[int]:
    raw = env.get(key, "")
    if raw == "":
        if required:
            raise RenderError(f"{key} is not set")
        return None
    try:
        return int(raw)
    except ValueError:
        raise RenderError(f"{key}={raw!r} is not an integer")


@dataclass
class Settings:
    model: str
    xp: int
    yd: int
    mode: str
    backend: str
    ips: List[str]
    gpus_per_node: int
    model_path: str
    prefill_tp: Optional[int]
    decode_tp: Optional[int]
    server_port: int
    router_port: int
    dist_init_port: int
    kv_ib_devices: str
    overlays: List[str]
    extra_all: List[str]
    extra_role: Dict[str, List[str]]
    extra_router: List[str]
    worker_startup_timeout: int

    @classmethod
    def from_env(cls, env: Mapping[str, str], nnodes: Optional[int] = None) -> "Settings":
        """Settings from an environment that has been through settings.sh."""
        model = env.get("MODEL_NAME", "")
        if not model:
            raise RenderError("MODEL_NAME is not set")
        xp, yd = _int(env, "xP"), _int(env, "yD")
        ips = [ip.strip() for ip in env.get("IPADDRS", "").split(",") if ip.strip()]
        if not ips:
            ips = [f"node{i}" for i in range(nnodes or xp + yd)]
        overlays = []
        agentic = env.get("BENCHMARK_SCRIPT") == "agentic"
        if agentic or env.get("ENABLE_RADIX_CACHE") == "1":
            overlays.append("radix_cache")
        if agentic or env.get("ENABLE_SERVER_METRICS") == "1":
            overlays.append("server_metrics")
        return cls(
            model=model,
            xp=xp,
            yd=yd,
            mode=env.get("PARALLEL_MODE", ""),
            backend=env.get("KV_TRANSFER_BACKEND", ""),
            ips=ips,
            gpus_per_node=_int(env, "GPUS_PER_NODE"),
            model_path=env.get("MODEL_PATH", "") or f"<models>/{model}",
            prefill_tp=_int(env, "PREFILL_TP_SIZE", required=False),
            decode_tp=_int(env, "DECODE_TP_SIZE", required=False),
            server_port=_int(env, "SERVER_PORT"),
            router_port=_int(env, "ROUTER_PORT"),
            dist_init_port=_int(env, "DIST_INIT_PORT"),
            kv_ib_devices=env.get("KV_IB_DEVICES", ""),
            overlays=overlays,
            extra_all=shlex.split(env.get("EXTRA_SERVER_ARGS", "")),
            extra_role={
                "prefill": shlex.split(env.get("EXTRA_PREFILL_ARGS", "")),
                "decode": shlex.split(env.get("EXTRA_DECODE_ARGS", "")),
            },
            extra_router=shlex.split(env.get("EXTRA_ROUTER_ARGS", "")),
            worker_startup_timeout=_int(env, "WORKER_STARTUP_TIMEOUT"),
        )


# --------------------------------------------------------------------------- topology

@dataclass
class Node:
    rank: int
    ip: str
    role: str                  # prefill | decode | router
    runs_router: bool = False
    tp: int = 0
    dp: int = 1
    ep: int = 1
    nnodes: int = 1            # nodes in this node's server
    node_rank: int = 0         # rank within that server
    leader_ip: str = ""
    server: int = 0            # index of the server within its role


@dataclass
class Topology:
    nodes: Dict[int, Node]
    leaders: Dict[str, List[str]] = field(default_factory=dict)
    dedicated_router: bool = False


def _role_sizes(s: Settings, role: str, role_nodes: int) -> Tuple[int, int, int, int]:
    """(tp, dp, ep, nodes per server) for one role."""
    gpus = s.gpus_per_node
    if s.mode == "wide_ep":
        size = role_nodes * gpus
        return size, size, size, role_nodes
    tp = (s.prefill_tp if role == "prefill" else s.decode_tp) or gpus
    if tp > gpus and tp % gpus:
        raise RenderError(f"{role} TP size {tp} is not a multiple of GPUS_PER_NODE={gpus}")
    return tp, 1, 1, max(1, tp // gpus)


def build_topology(s: Settings) -> Topology:
    if s.xp < 1 or s.yd < 1:
        raise RenderError(f"need at least one prefill and one decode node (xP={s.xp}, yD={s.yd})")
    n = len(s.ips)
    if n == s.xp + s.yd:
        first, dedicated = 0, False
    elif n == s.xp + s.yd + 1:
        first, dedicated = 1, True
    else:
        raise RenderError(
            f"{n} node(s) do not fit xP={s.xp} + yD={s.yd}: expected {s.xp + s.yd} "
            f"(router on node 0) or {s.xp + s.yd + 1} (dedicated router node)")
    topo = Topology(nodes={}, leaders={r: [] for r in ROLES}, dedicated_router=dedicated)
    if dedicated:
        topo.nodes[0] = Node(rank=0, ip=s.ips[0], role="router", runs_router=True)
    spans = {"prefill": list(range(first, first + s.xp)),
             "decode": list(range(first + s.xp, first + s.xp + s.yd))}
    for role, ranks in spans.items():
        tp, dp, ep, per = _role_sizes(s, role, len(ranks))
        if len(ranks) % per:
            raise RenderError(
                f"{len(ranks)} {role} node(s) cannot be split into servers of {per} node(s)")
        for index, start in enumerate(range(0, len(ranks), per)):
            group = ranks[start:start + per]
            leader = s.ips[group[0]]
            topo.leaders[role].append(leader)
            for i, rank in enumerate(group):
                topo.nodes[rank] = Node(
                    rank=rank, ip=s.ips[rank], role=role, runs_router=(rank == 0),
                    tp=tp, dp=dp, ep=ep, nnodes=len(group), node_rank=i,
                    leader_ip=leader, server=index)
    return topo


def validate(cat: Catalog, s: Settings) -> Topology:
    modes = cat.modes_for(s.model)
    if s.mode not in MODES:
        raise RenderError(f"PARALLEL_MODE={s.mode!r} is not one of {MODES}")
    if s.mode not in modes:
        raise RenderError(
            f"{s.model} does not support PARALLEL_MODE={s.mode} (supported: {', '.join(modes)}); "
            f"add a modes.{s.mode} block to its models.yaml entry to enable it")
    backend_modes = cat.backend(s.backend).get("modes") or list(MODES)
    if s.mode not in backend_modes:
        raise RenderError(
            f"KV_TRANSFER_BACKEND={s.backend} is not validated with PARALLEL_MODE={s.mode} "
            f"(models.yaml backends.{s.backend}.modes: {', '.join(backend_modes)})")
    for role in ROLES:
        cat.server_flags(s.model, s.mode, role, s.backend, s.overlays)
        cat.server_env(s.model, s.mode, role)
    return build_topology(s)


# --------------------------------------------------------------------------- commands

def server_argv(cat: Catalog, s: Settings, node: Node) -> List[str]:
    argv = ["--model-path", s.model_path, "--disaggregation-mode", node.role,
            "--host", node.ip, "--port", str(s.server_port), "--tp-size", str(node.tp)]
    if s.mode == "wide_ep":
        argv += ["--dp-size", str(node.dp), "--ep-size", str(node.ep)]
    if node.nnodes > 1 or s.mode == "wide_ep":
        argv += ["--dist-init-addr", f"{node.leader_ip}:{s.dist_init_port}",
                 "--nnodes", str(node.nnodes), "--node-rank", str(node.node_rank)]
    if s.kv_ib_devices:
        argv += ["--disaggregation-ib-device", s.kv_ib_devices]
    argv += flags_to_argv(cat.server_flags(s.model, s.mode, node.role, s.backend, s.overlays))
    return argv + s.extra_all + s.extra_role[node.role]


def router_argv(s: Settings, topo: Topology) -> List[str]:
    argv = ["--pd-disaggregation"]
    for ip in topo.leaders["prefill"]:
        argv += ["--prefill", f"http://{ip}:{s.server_port}"]
    for ip in topo.leaders["decode"]:
        argv += ["--decode", f"http://{ip}:{s.server_port}"]
    argv += ["--host", "0.0.0.0", "--port", str(s.router_port),
             "--worker-startup-timeout-secs", str(s.worker_startup_timeout),
             "--worker-startup-check-interval", "10"]
    return argv + s.extra_router


def describe(node: Node, mode: str) -> str:
    if node.role == "router":
        return "router"
    desc = (f"{node.role} server {node.server}, node {node.node_rank + 1}/{node.nnodes}, "
            f"tp={node.tp}" + (f" dp={node.dp} ep={node.ep}" if mode == "wide_ep" else ""))
    return desc + (" + router" if node.runs_router else "")


def node_vars(s: Settings, topo: Topology, node: Node) -> Dict[str, str]:
    return {
        "ROLE": node.role,
        "DESCRIPTION": describe(node, s.mode),
        "RUNS_ROUTER": "1" if node.runs_router else "0",
        "RUNS_SERVER": "0" if node.role == "router" else "1",
        "NODE_IP": node.ip,
        "ROUTER_IP": s.ips[0],
        # First prefill and decode server, where benchmarks read server metrics.
        "METRICS_ENDPOINTS": f"{topo.leaders['prefill'][0]}:{s.server_port} "
                             f"{topo.leaders['decode'][0]}:{s.server_port}",
    }


def _node(topo: Topology, rank: int) -> Node:
    if rank not in topo.nodes:
        raise RenderError(f"NODE_RANK={rank} is outside the {len(topo.nodes)}-node topology")
    return topo.nodes[rank]


def print_plan(cat: Catalog, s: Settings, topo: Topology, as_json: bool) -> None:
    rows = []
    for rank in sorted(topo.nodes):
        node = topo.nodes[rank]
        row = dict(asdict(node), description=describe(node, s.mode))
        if node.role != "router":
            row["env"] = cat.server_env(s.model, s.mode, node.role)
            row["server_cmd"] = ["python3", "-m", "sglang.launch_server"] + server_argv(cat, s, node)
        if node.runs_router:
            row["router_cmd"] = ["python3", "-m", "sglang_router.launch_router"] + router_argv(s, topo)
        rows.append(row)
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    where = "dedicated router node" if topo.dedicated_router else "router on node 0"
    print(f"model={s.model} mode={s.mode} backend={s.backend} xP={s.xp} yD={s.yd} "
          f"nodes={len(topo.nodes)} ({where}) overlays={','.join(s.overlays) or '-'}")
    for row in rows:
        print(f"\n[node {row['rank']}] {row['ip']}  {row['description']}")
        if row.get("env"):
            print("  env: " + " ".join(f"{k}={shlex.quote(v)}" for k, v in row["env"].items()))
        for key in ("server_cmd", "router_cmd"):
            if key in row:
                print("  " + shlex.join(row[key]))


def cmd_check(cat: Catalog) -> int:
    import sglang
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    known = {opt for action in parser._actions for opt in action.option_strings}
    used_flags: Dict[str, set] = {flag: {"launcher"} for flag in LAUNCHER_FLAGS}
    used_env: Dict[str, set] = {}
    for name in cat.names():
        for mode in cat.modes_for(name):
            for backend, spec in cat.backends.items():
                if mode not in ((spec or {}).get("modes") or MODES):
                    continue
                for role in ROLES:
                    where = f"{name}/{mode}/{role}"
                    for flag in cat.server_flags(name, mode, role, backend, list(cat.overlays)):
                        used_flags.setdefault(flag, set()).add(f"{where}/{backend}")
                    for var in cat.server_env(name, mode, role):
                        if var.startswith("SGLANG_"):
                            used_env.setdefault(var, set()).add(where)

    # SGLang reads its variables both through a registry and directly, so look
    # for each name anywhere in the installed package's source.
    unread = set(used_env)
    for path in Path(list(sglang.__path__)[0]).rglob("*.py"):
        text = path.read_text(errors="ignore")
        unread -= {var for var in unread if var in text}
        if not unread:
            break

    problems = [(flag, wheres, "flag this SGLang does not accept")
                for flag, wheres in used_flags.items() if flag not in known]
    problems += [(var, used_env[var], "variable this SGLang never reads") for var in unread]
    for item, wheres, what in sorted(problems):
        shown = ", ".join(sorted(wheres)[:4]) + (", ..." if len(wheres) > 4 else "")
        print(f"  {what}: {item}  (used by {len(wheres)}: {shown})")
    if problems:
        return 1
    print(f"OK: every catalog flag and SGLANG_* variable exists in this SGLang ({len(cat.names())} models)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=HERE / "models.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a run setting from the environment")
    ap.add_argument("--nnodes", type=int, help="node count when IPADDRS is unset (plan only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("models", "validate", "router-argv", "check"):
        sub.add_parser(name)
    sub.add_parser("plan").add_argument("--json", action="store_true")
    for name in ("node", "env", "argv"):
        sub.add_parser(name).add_argument("--rank", type=int, required=True)
    args = ap.parse_args(argv)

    env = dict(os.environ)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        env[key] = value

    try:
        cat = Catalog(args.models)
        if args.cmd == "models":
            for name in cat.names():
                print(f"{name}\t{','.join(cat.modes_for(name))}")
            return 0
        if args.cmd == "check":
            return cmd_check(cat)
        env = run_settings(env)
        s = Settings.from_env(env, args.nnodes)
        topo = validate(cat, s)
        if args.cmd == "plan":
            print_plan(cat, s, topo, args.json)
        elif args.cmd == "node":
            for key, value in node_vars(s, topo, _node(topo, args.rank)).items():
                print(f"{key}={shlex.quote(value)}")
        elif args.cmd == "env":
            node = _node(topo, args.rank)
            if node.role != "router":
                for key, value in cat.server_env(s.model, s.mode, node.role).items():
                    if key not in env:
                        print(f"export {key}={shlex.quote(value)}")
        elif args.cmd == "argv":
            node = _node(topo, args.rank)
            if node.role == "router":
                raise RenderError(f"node {args.rank} runs no server")
            sys.stdout.write("".join(a + "\0" for a in server_argv(cat, s, node)))
        elif args.cmd == "router-argv":
            sys.stdout.write("".join(a + "\0" for a in router_argv(s, topo)))
        return 0
    except RenderError as exc:
        print(f"render.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
