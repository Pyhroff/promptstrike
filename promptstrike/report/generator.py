"""Generates HTML reports from campaign data."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_OWASP_LABELS = {
    "LLM01": "Prompt Injection",
    "LLM02": "Insecure Output Handling",
    "LLM03": "Training Data Poisoning",
    "LLM06": "Sensitive Information Disclosure",
    "LLM07": "Insecure Plugin Design",
    "LLM09": "Overreliance",
}

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_html_report(campaign: dict, runs: list[dict]) -> str:
    succeeded  = [r for r in runs if r["status"] == "success"]
    failed     = [r for r in runs if r["status"] != "success"]
    asr        = len(succeeded) / max(1, len(runs)) * 100

    owasp_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "succeeded": 0})
    for r in runs:
        key = r.get("owasp") or "Unknown"
        owasp_counts[key]["total"] += 1
        if r["status"] == "success":
            owasp_counts[key]["succeeded"] += 1

    owasp_breakdown = [
        {
            "code": k,
            "label": _OWASP_LABELS.get(k, "Other"),
            "total": v["total"],
            "succeeded": v["succeeded"],
            "asr": v["succeeded"] / max(1, v["total"]) * 100,
        }
        for k, v in sorted(owasp_counts.items())
    ]

    cat_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "succeeded": 0})
    for r in runs:
        key = r.get("category") or "unknown"
        cat_counts[key]["total"] += 1
        if r["status"] == "success":
            cat_counts[key]["succeeded"] += 1

    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=True)
    template = env.get_template("report.html")

    return template.render(
        campaign=campaign,
        runs=runs,
        succeeded=succeeded,
        failed=failed,
        asr=round(asr, 1),
        owasp_breakdown=owasp_breakdown,
        cat_counts=dict(cat_counts),
        total=len(runs),
    )


def generate_sweep_report(sweep_name: str, results: list[dict], algo: str = "pair") -> str:
    """
    results: list of {target, campaign_id, succeeded, total, asr, calls_used, runs}
    """
    # Unique categories across all runs
    categories = sorted({
        r.get("category") or "unknown"
        for item in results
        for r in item["runs"]
    })

    # matrix[category][target] = {succeeded, total, asr}
    matrix: dict[str, dict[str, dict]] = {}
    for cat in categories:
        matrix[cat] = {}
        for item in results:
            cat_runs = [r for r in item["runs"] if (r.get("category") or "unknown") == cat]
            cat_ok   = sum(1 for r in cat_runs if r["status"] == "success")
            matrix[cat][item["target"]] = {
                "succeeded": cat_ok,
                "total": len(cat_runs),
                "asr": round(cat_ok / max(1, len(cat_runs)) * 100) if cat_runs else None,
            }

    all_jailbreaks = sorted(
        [
            {**r, "model": item["target"]}
            for item in results
            for r in item["runs"]
            if r["status"] == "success"
        ],
        key=lambda x: x.get("final_score", 0),
        reverse=True,
    )

    total_jailbreaks = len(all_jailbreaks)
    most_vulnerable  = results[0]["target"] if results else "—"

    env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=True)
    template = env.get_template("sweep_report.html")

    return template.render(
        sweep_name=sweep_name,
        algo=algo,
        results=results,
        categories=categories,
        matrix=matrix,
        all_jailbreaks=all_jailbreaks[:9],
        total_targets=len(results),
        total_behaviors=results[0]["total"] if results else 0,
        total_jailbreaks=total_jailbreaks,
        most_vulnerable=most_vulnerable,
        most_vulnerable_asr=results[0]["asr"] if results else 0,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    )
