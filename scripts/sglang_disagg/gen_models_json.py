#!/usr/bin/env python3
"""Generate models.json (madengine cards) from matrix.yaml.

Every card is validated against models.yaml with render.py, so a card can only
name a model, mode and backend the catalog supports.

  python3 gen_models_json.py           rewrite models.json
  python3 gen_models_json.py --check   exit 1 if models.json is out of date
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from render import Catalog, RenderError, Settings, run_settings, validate  # noqa: E402


def build_cards(matrix: dict, catalog: Catalog) -> list:
    cards = []
    for group in matrix["groups"]:
        variant = group["variant"]
        for model in group["models"]:
            name = "_".join(filter(None, [
                "pyt_sglang_disagg", variant, group.get("suffix"), matrix["slugs"][model]]))
            card = {"name": name}
            card.update(copy.deepcopy(matrix["card"]))
            card["tags"] = [t.format(variant=variant) for t in card["tags"]] + group.get("tags", [])
            xp, yd = group.get("xP", 1), group.get("yD", 1)
            card["distributed"] = {"launcher": "slurm_multi", "nnodes": xp + yd}
            env = card["env_vars"]
            env.update({"MODEL_NAME": model, "xP": str(xp), "yD": str(yd)})
            env.update({k: str(v) for k, v in matrix["variants"][variant].items()})
            env.update({k: str(v) for k, v in group.get("env_vars", {}).items()})
            try:
                validate(catalog, Settings.from_env(run_settings(env), nnodes=xp + yd))
            except RenderError as exc:
                raise SystemExit(f"gen_models_json.py: card {name}: {exc}")
            cards.append(card)
    names = [c["name"] for c in cards]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SystemExit(f"gen_models_json.py: duplicate card names: {', '.join(sorted(dupes))}")
    return cards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if models.json is out of date")
    ap.add_argument("--matrix", type=Path, default=HERE / "matrix.yaml")
    ap.add_argument("--output", type=Path, default=HERE / "models.json")
    args = ap.parse_args()

    catalog = Catalog(HERE / "models.yaml")
    cards = build_cards(yaml.safe_load(args.matrix.read_text()), catalog)
    text = json.dumps(cards, indent=4) + "\n"
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is out of date; run: python3 {Path(__file__).name}", file=sys.stderr)
            return 1
        print(f"{args.output.name} is up to date ({len(cards)} cards)")
        return 0
    args.output.write_text(text)
    print(f"wrote {len(cards)} cards to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
