#!/usr/bin/env python3
"""Offline tests for the recipe: python3 tests/test_render.py (no GPUs or cluster needed)."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

RECIPE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RECIPE))
sys.dont_write_bytecode = True

import render  # noqa: E402
from render import (Catalog, RenderError, Settings, build_topology, node_vars,  # noqa: E402
                    router_argv, run_settings, server_argv, validate)

CATALOG = Catalog(RECIPE / "models.yaml")


def settings(**env):
    base = {"MODEL_NAME": "Llama-3.1-8B-Instruct", "MODEL_PATH": "/m"}
    base.update({k: str(v) for k, v in env.items()})
    return Settings.from_env(run_settings(base))


def flags(argv):
    out, i = {}, 0
    while i < len(argv):
        key, vals = argv[i], []
        i += 1
        while i < len(argv) and not argv[i].startswith("--"):
            vals.append(argv[i])
            i += 1
        out[key] = vals
    return out


def catalog_from(models_yaml):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "models.yaml").write_text(textwrap.dedent(models_yaml)
                                     + "\nbackends: {mori: {args: {--disaggregation-transfer-backend: mori}}}\n")
    return Catalog(tmp / "models.yaml")


class MergeTest(unittest.TestCase):
    def test_layer_order_and_removal(self):
        cat = catalog_from("""
            defaults:
              base: {--a: 1, --b: 1, --gone: true}
              roles: {decode: {tp: {--b: 2}}}
            models:
              M:
                base: {--a: 3, --gone: false}
                roles: {decode: {--c: x, tp: {--c: y}}}
            """)
        got = cat.server_flags("M", "tp", "decode", "mori")
        # Model beats defaults even at a less specific level; role x mode beats role.
        self.assertEqual(got, {"--a": 3, "--b": 2, "--c": "y",
                               "--disaggregation-transfer-backend": "mori"})

    def test_extends_null_drops_inherited_value(self):
        r1 = CATALOG.server_flags("DeepSeek-R1", "wide_ep", "prefill", "mori")
        v3 = CATALOG.server_flags("DeepSeek-V3", "wide_ep", "prefill", "mori")
        self.assertEqual(v3["--mem-fraction-static"], 0.65)
        self.assertEqual(r1["--mem-fraction-static"], 0.73)
        self.assertEqual({k: v for k, v in r1.items() if k != "--mem-fraction-static"},
                         {k: v for k, v in v3.items() if k != "--mem-fraction-static"})

    def test_values(self):
        argv = render.flags_to_argv({"--x": True, "--r": "1..3", "--l": [4, 5], "--f": 0.5, "--n": None})
        self.assertEqual(argv, ["--x", "--r", "1", "2", "3", "--l", "4", "5", "--f", "0.5"])

    def test_overlays(self):
        s = settings(BENCHMARK_SCRIPT="agentic")
        decode = CATALOG.server_flags(s.model, "tp", "decode", "mori", s.overlays)
        self.assertNotIn("--disable-radix-cache", decode)
        self.assertTrue(decode["--enable-metrics"])

    def test_catalog_rejects_launcher_flags_and_typos(self):
        with self.assertRaisesRegex(RenderError, "set by the launcher"):
            catalog_from("models: {M: {base: {--tp-size: 4}}}").server_flags("M", "tp", "prefill", "mori")
        with self.assertRaisesRegex(RenderError, "neither a --flag, a mode nor env"):
            catalog_from("models: {M: {roles: {decode: {tpp: {--x: 1}}}}}").server_flags("M", "tp", "decode", "mori")


class EnvTest(unittest.TestCase):
    def test_env_layers_like_flags(self):
        cat = catalog_from("""
            defaults:
              env: {A: "1", B: "1"}
              modes: {wide_ep: {env: {B: "2", C: "1"}}}
            models:
              M:
                modes: {wide_ep: {}}
                env: {A: "3"}
                roles: {decode: {wide_ep: {env: {C: null, D: "4"}}}}
            """)
        self.assertEqual(cat.server_env("M", "wide_ep", "decode"), {"A": "3", "B": "2", "D": "4"})
        self.assertEqual(cat.server_env("M", "tp", "decode"), {"A": "3", "B": "1"})

    def test_mori_tuning_only_where_it_applies(self):
        tp = CATALOG.server_env("DeepSeek-V3", "tp", "decode")
        prefill = CATALOG.server_env("DeepSeek-V3", "wide_ep", "prefill")
        decode = CATALOG.server_env("DeepSeek-V3", "wide_ep", "decode")
        self.assertFalse(any(k.startswith(("SGLANG_MORI", "MORI_")) for k in tp))
        self.assertEqual(prefill["MORI_SHMEM_HEAP_SIZE"], "17179869184")
        self.assertNotIn("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", prefill)
        self.assertEqual(decode["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"], "160")

    def test_env_command_lets_the_environment_win(self):
        env = dict(os.environ, MODEL_NAME="DeepSeek-V3", PARALLEL_MODE="wide_ep", MODEL_PATH="/m",
                   IPADDRS="a,b", MORI_SHMEM_HEAP_SIZE="1")
        out = subprocess.run([sys.executable, str(RECIPE / "render.py"), "env", "--rank", "1"],
                             env=env, capture_output=True, text=True, check=True).stdout
        self.assertIn("export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160", out)
        self.assertNotIn("MORI_SHMEM_HEAP_SIZE", out)


class SettingsTest(unittest.TestCase):
    def test_defaults_and_older_names_come_from_settings_sh(self):
        env = run_settings({"MODEL_NAME": "M", "DP_MODE": "1", "RUN_MORI": "0", "SKIP_BENCHMARK": "1"})
        self.assertEqual((env["PARALLEL_MODE"], env["KV_TRANSFER_BACKEND"], env["BENCHMARK_SCRIPT"]),
                         ("wide_ep", "mooncake", "none"))
        self.assertEqual((env["xP"], env["ROUTER_PORT"], env["WORKER_STARTUP_TIMEOUT"]), ("1", "2322", "3600"))

    def test_cluster_profile_selection(self):
        def profile(host, **env):
            script = ('hostname() { echo "$HOST"; }; source settings.sh; load_cluster_profile . || exit 3; '
                      'echo "$CLUSTER $KV_IB_DEVICES $MODEL_ROOTS"')
            clean = {"PATH": os.environ["PATH"], "HOME": "/h", "USER": "u", "HOST": host, **env}
            out = subprocess.run(["bash", "-c", script], cwd=RECIPE, env=clean, capture_output=True, text=True)
            return out.returncode, out.stdout.split()

        self.assertEqual(profile("useocpm2m-097-030")[1][:2], ["oci-m2m", "mlx5_1"])
        self.assertEqual(profile("useocpm2m-097-030", USE_CX7_NICS="1")[1][1].count("mlx5_"), 8)
        self.assertEqual(profile("useocpm2m-097-030", MODEL_ROOTS="/fast")[1][2], "/fast")
        self.assertEqual(profile("elsewhere")[1], ["none", "/models:/h/models"])
        self.assertEqual(profile("x", CLUSTER="nope")[0], 3)


class TopologyTest(unittest.TestCase):
    def test_colocated_router(self):
        s = settings(IPADDRS="a,b,c", xP=1, yD=2)
        topo = build_topology(s)
        self.assertEqual([topo.nodes[r].role for r in range(3)], ["prefill", "decode", "decode"])
        self.assertTrue(topo.nodes[0].runs_router)
        self.assertEqual(topo.leaders, {"prefill": ["a"], "decode": ["b", "c"]})

    def test_dedicated_router(self):
        topo = build_topology(settings(IPADDRS="r,p,d"))
        self.assertEqual(topo.nodes[0].role, "router")
        self.assertEqual(router_argv(settings(IPADDRS="r,p,d"), topo)[:5],
                         ["--pd-disaggregation", "--prefill", "http://p:3000", "--decode", "http://d:3000"])

    def test_wide_ep_spans_role_nodes(self):
        s = settings(MODEL_NAME="DeepSeek-V3", PARALLEL_MODE="wide_ep", xP=2, yD=1, IPADDRS="a,b,c")
        topo = validate(CATALOG, s)
        f = flags(server_argv(CATALOG, s, topo.nodes[1]))
        self.assertEqual(f["--tp-size"], ["16"])
        self.assertEqual(f["--dist-init-addr"], ["a:5757"])
        self.assertEqual(f["--node-rank"], ["1"])
        self.assertEqual(topo.leaders["prefill"], ["a"])
        self.assertEqual(node_vars(s, topo, topo.nodes[1])["DESCRIPTION"],
                         "prefill server 0, node 2/2, tp=16 dp=16 ep=16")

    def test_multi_node_tp(self):
        s = settings(IPADDRS="a,b,c,d", xP=2, yD=2, PREFILL_TP_SIZE=16)
        topo = build_topology(s)
        self.assertEqual(topo.leaders["prefill"], ["a"])
        self.assertEqual(topo.leaders["decode"], ["c", "d"])
        self.assertEqual(flags(server_argv(CATALOG, s, topo.nodes[1]))["--nnodes"], ["2"])

    def test_rejects(self):
        with self.assertRaisesRegex(RenderError, "do not fit"):
            build_topology(settings(IPADDRS="a,b,c,d"))
        with self.assertRaisesRegex(RenderError, "does not support PARALLEL_MODE=wide_ep"):
            validate(CATALOG, settings(PARALLEL_MODE="wide_ep"))
        with self.assertRaisesRegex(RenderError, "not validated with"):
            validate(CATALOG, settings(MODEL_NAME="DeepSeek-V3", PARALLEL_MODE="wide_ep",
                                       KV_TRANSFER_BACKEND="mooncake"))

    def test_extra_args_come_last(self):
        s = settings(IPADDRS="a,b", EXTRA_DECODE_ARGS="--mem-fraction-static 0.5")
        argv = server_argv(CATALOG, s, build_topology(s).nodes[1])
        self.assertEqual(argv[-2:], ["--mem-fraction-static", "0.5"])


class CardsTest(unittest.TestCase):
    def test_models_json_is_current(self):
        result = subprocess.run([sys.executable, str(RECIPE / "gen_models_json.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_card_renders(self):
        for card in json.loads((RECIPE / "models.json").read_text()):
            env = run_settings(dict(card["env_vars"], MODEL_PATH="/m"))
            s = Settings.from_env(env, nnodes=card["distributed"]["nnodes"])
            topo = validate(CATALOG, s)
            for node in topo.nodes.values():
                if node.role != "router":
                    self.assertTrue(server_argv(CATALOG, s, node), card["name"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
