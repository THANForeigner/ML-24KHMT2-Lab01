"""HTTP smoke test for a running showcase; standard library only.

Run after `docker compose up --build -d --wait`:
    python scripts/smoke_web.py --url http://127.0.0.1:8080
"""
import argparse
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    def call(path, body=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        payload = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(base + path, data=payload, headers=headers), timeout=90) as response:
            return response.read()

    def api(path, body=None):
        return json.loads(call(path, body))

    assert api("/api/health")["status"] == "ok"
    html = call("/").decode()
    asset = re.search(r'src="(/assets/[^"]+\.js)"', html)
    assert asset, "Built React asset not found in HTML"
    assert len(call(asset.group(1))) > 1000, "React asset missing"
    print("[PASS] Health and built React assets")

    bootstrap = api("/api/bootstrap")
    names = ["Simple", "Interaction", "Piecewise", "Polynomial", "Ridge", "Elastic Net"]
    assert [model["id"] for model in bootstrap["models"]] == names
    assert all(model["available"] for model in bootstrap["models"]), "Missing model artifact"
    assert len(bootstrap["fields"]) == len(bootstrap["profile"]) == 12
    profile = bootstrap["profile"]
    expected = [2.650811, 2.698924, 2.644931, 2.562398, 2.647915, 2.684647]
    comparison = api("/api/compare", {"profile": profile, "variant": "M03"})
    assert [row["model"] for row in comparison["rows"]] == names
    for row, want in zip(comparison["rows"], expected):
        assert row["error"] is None, f"{row['model']}: {row['error']}"
        assert math.isclose(row["gpa"], want, abs_tol=.000001), row
    print("[PASS] All six real estimators and expected predictions")

    for variant, field in bootstrap["simple_variants"].items():
        prediction = api("/api/predict", {
            "model": "Simple", "variant": variant, "profile": profile, "variable": field,
        })
        assert math.isfinite(prediction["gpa"])
        assert prediction["variant"] == variant
        assert any(point["x"] == profile[field] for point in prediction["curve"])
    print("[PASS] Simple M01–M05 and sensitivity curves")

    validation = api("/api/validation")
    assert validation["n_rows"] == 7500
    scores = {row["model"]: row for row in validation["metrics"]}
    assert not scores["Simple"]["eligible"] and scores["Simple"]["rmse"] is None
    assert sum(row["eligible"] for row in scores.values()) == 5
    assert math.isclose(scores["Ridge"]["rmse"], .16076058372183918, abs_tol=1e-8)
    assert math.isclose(scores["Piecewise"]["rmse"], .14671197266237843, abs_tol=1e-8)
    agreement = api("/api/agreement?model=Ridge")
    assert len(agreement["points"]) == 800 and agreement["n_rows"] == 7500
    assert agreement["domain"][1] > 4, "Raw axis domain was clipped"
    print("[PASS] Validation metrics, exclusion policy and real scatter data")

    try:
        api("/api/compare", {"profile": {}, "variant": "M03"})
    except HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError("Empty profile was accepted")
    for private_path in ["/.env", "/data/train/dataset.csv"]:
        try:
            call(private_path)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError(f"Private file exposed: {private_path}")
    print("[PASS] Input validation and private-file boundaries")
    print(f"Showcase ready: {base}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"[FAIL] {exc}") from exc
