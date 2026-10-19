"""Validate the downloaded CoLight/CityFlow benchmark files and record provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nashgat.envs.roadnet import load_flows, load_roadnet, validate  # noqa: E402

FILES = {
    "jinan_3x4": ("data/raw/colight/data/Jinan/3_4/roadnet_3_4.json",
                  ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json", "anon_3_4_jinan_real_2500.json"]),
    "hangzhou_4x4": ("data/raw/colight/data/Hangzhou/4_4/roadnet_4_4.json",
                     ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5734.json",
                      "anon_4_4_hangzhou_real_5816.json"]),
    "newyork_28x7": ("data/raw/colight/data/NewYork/28_7/roadnet_28_7.json",
                     ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    report = {"source": "https://github.com/wingsweihua/colight (data/), sparse clone", "networks": {}}
    for name, (rn, flows) in FILES.items():
        rnp = ROOT / rn
        net = load_roadnet(rnp)
        entry = {"roadnet": rn, "roadnet_sha256": sha256(rnp), "flows": {}}
        for f in flows:
            fp = rnp.parent / f
            v = validate(net, load_flows(fp))
            v["sha256"] = sha256(fp)
            entry["flows"][f] = v
        report["networks"][name] = entry
    out = ROOT / "results/dataset_validation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    main()
