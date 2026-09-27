"""
Master E2E Test Runner for Dual-Site Blood Pressure Estimation Framework
Author: test_writer_1 (E2E Testing Architect)

Orchestrates execution of all 4 test tiers and the Diagnostic Template Collapse Audit:
- Tier 1: Isolated Feature Coverage (75 tests across 15 features)
- Tier 2: Boundary & Corner Cases (75 tests across 15 features)
- Tier 3: Pairwise Cross-Feature Combinations (16 tests)
- Tier 4: Real-World Clinical and Mobile Workloads (8 tests)
- Diagnostic Template Collapse Audit (120 subjects)
Total: 174 test cases + 1 forensic audit
"""

import sys
import os
import time
import argparse
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

TIER_FILES = {
    "tier1": "tests/e2e/test_tier1_feature.py",
    "tier2": "tests/e2e/test_tier2_boundary.py",
    "tier3": "tests/e2e/test_tier3_pairwise.py",
    "tier4": "tests/e2e/test_tier4_workloads.py",
}


def run_command(cmd: list) -> tuple:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
    duration = time.perf_counter() - t0
    return res.returncode == 0, res.stdout, res.stderr, duration


def run_tier(tier_name: str, file_path: str):
    print(f"\n>> Executing {tier_name.upper()} ({file_path})...")
    cmd = [sys.executable, "-m", "pytest", file_path, "-v", "--tb=short"]
    success, stdout, stderr, duration = run_command(cmd)

    # Parse passed/failed counts from stdout
    passed = 0
    failed = 0
    for line in stdout.splitlines():
        if "passed in" in line or "failed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed":
                    try:
                        passed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                elif p == "failed":
                    try:
                        failed = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass

    status_str = "PASSED" if success else "FAILED"
    print(f"[{status_str}] {tier_name.upper()}: {passed} passed, {failed} failed in {duration:.2f}s")
    if not success:
        print(stderr or stdout)

    return {
        "tier": tier_name,
        "file": file_path,
        "success": success,
        "passed": passed,
        "failed": failed,
        "duration_sec": round(duration, 3)
    }


def run_audit():
    print("\n>> Executing Diagnostic Template Collapse Audit...")
    cmd = [sys.executable, "tests/audit/template_collapse_audit.py"]
    success, stdout, stderr, duration = run_command(cmd)

    status_str = "PASSED" if success else "FAILED"
    print(f"[{status_str}] TEMPLATE COLLAPSE AUDIT finished in {duration:.2f}s")
    if not success:
        print(stderr or stdout)
    else:
        # Print summary lines
        for line in stdout.splitlines():
            if "Pearson Correlation" in line or "Variance Ratio" in line or "Mean Absolute Error" in line or "AAMI" in line or "BHS" in line or "SUCCESS" in line:
                print("   " + line)

    return {
        "tier": "audit",
        "file": "tests/audit/template_collapse_audit.py",
        "success": success,
        "duration_sec": round(duration, 3)
    }


def main():
    parser = argparse.ArgumentParser(description="Master E2E Test Runner for Dual-Site BP Estimation Framework")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run a specific tier (1, 2, 3, or 4)")
    parser.add_argument("--audit", action="store_true", help="Run only the Template Collapse Audit")
    parser.add_argument("--all", action="store_true", default=True, help="Run all tiers and the audit (default)")
    args = parser.parse_args()

    print("=" * 80)
    print("MASTER E2E TEST RUNNER — DUAL-SITE BLOOD PRESSURE ESTIMATION FRAMEWORK")
    print(f"Project Root: {PROJECT_ROOT}")
    print("Target Device: Xiaomi Poco M6 Pro 5G (Qualcomm Snapdragon 4 Gen 2)")
    print("=" * 80)

    results = []
    all_passed = True

    if args.tier:
        tier_key = f"tier{args.tier}"
        res = run_tier(tier_key, TIER_FILES[tier_key])
        results.append(res)
        all_passed = res["success"]
    elif args.audit:
        res = run_audit()
        results.append(res)
        all_passed = res["success"]
    else:
        # Run all tiers 1 to 4
        for tier_key, file_path in TIER_FILES.items():
            res = run_tier(tier_key, file_path)
            results.append(res)
            if not res["success"]:
                all_passed = False

        # Run template collapse audit
        audit_res = run_audit()
        results.append(audit_res)
        if not audit_res["success"]:
            all_passed = False

    # Print Master Summary Table
    print("\n" + "=" * 80)
    print("SUMMARY OF E2E TEST EXECUTION")
    print("=" * 80)
    print(f"{'Tier / Suite':<30} | {'Status':<10} | {'Passed':<8} | {'Failed':<8} | {'Duration':<10}")
    print("-" * 80)

    total_passed = 0
    total_failed = 0
    total_time = 0.0

    for r in results:
        p = r.get("passed", "N/A")
        f = r.get("failed", "N/A")
        if isinstance(p, int):
            total_passed += p
        if isinstance(f, int):
            total_failed += f
        total_time += r["duration_sec"]
        status = "PASS" if r["success"] else "FAIL"
        print(f"{r['tier'].upper():<30} | {status:<10} | {str(p):<8} | {str(f):<8} | {r['duration_sec']:.2f}s")

    print("-" * 80)
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"{'TOTAL':<30} | {overall_status:<10} | {total_passed:<8} | {total_failed:<8} | {total_time:.2f}s")
    print("=" * 80)

    # Save JSON summary artifact
    summary_path = PROJECT_ROOT / "test_results_summary.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_status": overall_status,
        "all_passed": all_passed,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "total_duration_sec": round(total_time, 2),
        "results": results
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nArtifact saved: {summary_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
