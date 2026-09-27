#!/usr/bin/env python3
"""One-command packager for the ML Challenge 2026 submission zip.

Writes <team>_submission.zip to the repo root with EXACTLY (nothing else):
  output/{matching_results.tsv,candidate_pairs.tsv},
  code/business_entity_resolution/{src/**, README.md, requirements.txt},
  Documentation_template.md (the filled template, at the zip root).

  e2e   rerun kaggle/run_smoke.py (in-process, same .venv interpreter, all
        gates), package the tiny TSVs as team 'team', run the official
        validator, assert the zip structure; exit 0 only if everything holds.
  real  --zip KAGGLE_RETURN.ZIP --team NAME : package the real downloaded
        kaggle_return.zip (Stage H stores the TSVs under output/ inside it —
        verified in cell "S8 — Stage H"). Run in the final minutes.

Stdlib only. Spawns no subprocess: run_smoke.py and the validator are imported
and called in-process (same checks; satisfies the Mimosa write gate that
rejects variable-argv subprocess calls).
"""
import argparse, importlib.util, os, re, shutil, sys, tempfile, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SMOKE = os.path.join(ROOT, "kaggle", "run_smoke.py")
VALIDATOR = os.path.join(ROOT, "student_resource", "utils", "validate_submission.py")
DOC = os.path.join(ROOT, "student_resource", "Documentation_template.md")
CODE_DIR = os.path.join(ROOT, "code", "business_entity_resolution")
TEST_TINY = os.path.join(ROOT, "work_tiny", "dataset", "test")
TEST_REAL = os.path.join(ROOT, "student_resource", "dataset", "test")
# Smoke TSVs: run_smoke.py:9,21 sets BER_WORK=<ROOT>/work_tiny/work_wfgate; notebook
# cell S0a derives OUTPUT_DIR = dirname(WORK)/output there (confirmed on disk).
SMOKE_OUTPUT = os.path.join(ROOT, "work_tiny", "output")
MATCHING, CANDIDATE = "matching_results.tsv", "candidate_pairs.tsv"
RETURN_ZIP_MATCHING, RETURN_ZIP_CANDIDATE = ("output/" + MATCHING,
                                             "output/" + CANDIDATE)
TOP_LEVEL = {"output", "code", "Documentation_template.md"}


def log(msg):
    print(msg, flush=True)


def fail(msg):
    log("GATE FAIL: " + msg)
    sys.exit(1)


def load_module(name, path):
    """Import a repo script (it has an __main__ guard) as a module."""
    if not os.path.isfile(path):
        fail("missing " + path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def collect_src_files():
    """(abs_path, zip_arc) per src file; skips hidden/__pycache__/.pyc junk."""
    src = os.path.join(CODE_DIR, "src")
    if not os.path.isdir(src):
        fail("missing " + src)
    out = []
    for base, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for f in sorted(files):
            if f.endswith((".pyc", ".pyo")) or f.startswith("."):
                continue
            p = os.path.join(base, f)
            out.append((p, "code/business_entity_resolution/"
                        + os.path.relpath(p, CODE_DIR).replace(os.sep, "/")))
    if not out:
        fail("no source files found under " + src)
    return sorted(out, key=lambda t: t[1])


def check_doc_filled():
    """GATE: no FILL markers left in the template; warn on bracket placeholders."""
    if not os.path.isfile(DOC):
        fail("missing " + DOC)
    with open(DOC, encoding="utf-8") as f:
        lines = f.read().splitlines()
    bad = [l for l in lines if "FILL" in l]
    if bad:
        log("=" * 70)
        log("WARNING - Documentation_template.md still contains FILL markers:")
        for l in bad:
            log("  | " + l.strip()[:110])
        log("THE METHODOLOGY DOC MUST BE FILLED BEFORE SUBMITTING.")
        log("=" * 70)
        fail("%d FILL marker(s) left in Documentation_template.md" % len(bad))
    for pat in ("[Your Team Name]", "[List all team members]", "[Date]"):
        if any(pat in l for l in lines):
            log("REMINDER (not a gate): doc still contains the placeholder %r - "
                "fill it before the real submission." % pat)
            break
    log("[doc] no FILL markers in Documentation_template.md")


def assemble(team, output_dir):
    """Build <team>_submission.zip at repo root from a dir holding the two TSVs."""
    src_files = collect_src_files()
    check_doc_filled()
    for name in (MATCHING, CANDIDATE):
        p = os.path.join(output_dir, name)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            fail("missing or empty: " + p)
        log("[assemble] input %s: %d bytes" % (p, os.path.getsize(p)))
    for rel in ("README.md", "requirements.txt"):
        if not os.path.isfile(os.path.join(CODE_DIR, rel)):
            fail("missing code/business_entity_resolution/" + rel)
    zip_path = os.path.join(ROOT, "%s_submission.zip" % team)
    log("[assemble] writing " + zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in (MATCHING, CANDIDATE):
            zf.write(os.path.join(output_dir, name), "output/" + name)
        log("[assemble] + output/{%s,%s}" % (MATCHING, CANDIDATE))
        for rel in ("README.md", "requirements.txt"):
            zf.write(os.path.join(CODE_DIR, rel),
                     "code/business_entity_resolution/" + rel)
        log("[assemble] + code/business_entity_resolution/{README.md,requirements.txt}")
        for p, arc in src_files:
            zf.write(p, arc)
        log("[assemble] + %d files under code/business_entity_resolution/src/"
            % len(src_files))
        with open(DOC, "rb") as f:
            zf.writestr("Documentation_template.md", f.read())
        log("[assemble] + Documentation_template.md")
    return zip_path, src_files


def check_zip(zip_path, output_dir, src_files):
    """GATE: exact structure, no junk, TSV bytes intact, doc clean, CRC ok."""
    log("[check] inspecting " + zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        tops = {n.split("/", 1)[0] for n in names}
        expected = {"/".join(("output", MATCHING)), "/".join(("output", CANDIDATE)),
                    "code/business_entity_resolution/README.md",
                    "code/business_entity_resolution/requirements.txt",
                    "Documentation_template.md"} | {a for _, a in src_files}
        junky = [n for n in names
                 if any(p == "__pycache__" or p.startswith(".") for p in n.split("/"))
                 or n.endswith((".pyc", ".pyo"))]
        if tops != TOP_LEVEL:
            fail("top-level entries %s != expected %s"
                 % (sorted(tops), sorted(TOP_LEVEL)))
        for label, bad in [("duplicate zip entries",
                            sorted({n for n in names if names.count(n) > 1})),
                           ("missing required entries",
                            sorted(expected - set(names))),
                           ("junk entries present", sorted(set(names) - expected)),
                           ("hidden/__pycache__/.pyc junk entries", sorted(junky))]:
            if bad:
                fail("%s: %s" % (label, bad))
        for name in (MATCHING, CANDIDATE):
            with open(os.path.join(output_dir, name), "rb") as f:
                if zf.read("output/" + name) != f.read():
                    fail("output/%s in the zip differs from the source file" % name)
        if "FILL" in zf.read("Documentation_template.md").decode("utf-8"):
            fail("packaged Documentation_template.md still has FILL markers")
        if zf.testzip() is not None:
            fail("zip CRC check failed")
    log("[check] OK: %d entries; top-level %s; no junk; TSV bytes verified; CRC ok"
        % (len(names), sorted(tops)))


def run_validator(matching, candidate, test_dir, gate):
    """Run the official validator in-process (imported, same checks as its CLI).

    Listed errors = hard fail in both modes; an environmental crash (e.g. OOM on
    the full candidate set) fails e2e but only warns in real mode.
    """
    if not os.path.isfile(os.path.join(test_dir, "test_source1.tsv")):
        if gate:
            fail("test_source1.tsv not found in " + test_dir)
        log("WARNING: no test_source1.tsv in %s - skipping validator." % test_dir)
        return
    mod = load_module("ber_validate_submission", VALIDATOR)
    log("[validate] validate_submission.validate(matching=%s, candidate=%s, "
        "test_dir=%s)" % (matching, candidate, test_dir))
    try:
        errors, warnings = mod.validate(matching, candidate, test_dir,
                                        check_ids=False)
    except Exception as e:  # environmental failure, not a formatting finding
        if gate:
            fail("validator crashed: %r" % (e,))
        log("WARNING: validator crashed (%r), likely memory on the full candidate "
            "set - re-run it manually with --matching only." % (e,))
        return
    for w in warnings:
        log("WARNING: " + w)
    if errors:
        for i, e in enumerate(errors, 1):
            log("  %d. %s" % (i, e))
        fail("%d validator issue(s) to fix before submitting" % len(errors))
    log("[validate] PASS - no blocking issues found")


def locate_smoke_tsvs():
    """Dir holding the smoke run's TSVs; expected path first, then a scan."""
    if all(os.path.isfile(os.path.join(SMOKE_OUTPUT, n))
           for n in (MATCHING, CANDIDATE)):
        return SMOKE_OUTPUT
    log("[e2e] TSVs not in %s - searching under work_tiny/ ..." % SMOKE_OUTPUT)
    hits = {}
    for base, _dirs, files in os.walk(os.path.join(ROOT, "work_tiny")):
        for n in (MATCHING, CANDIDATE):
            if n in files and n not in hits:
                hits[n] = base
    if set(hits) == {MATCHING, CANDIDATE} and hits[MATCHING] == hits[CANDIDATE]:
        log("[e2e] found smoke TSVs in " + hits[MATCHING])
        return hits[MATCHING]
    fail("smoke output TSVs not found under work_tiny/ - did run_smoke.py pass?")


def cmd_e2e(_args):
    log("[e2e] step 1/4: regenerate tiny outputs - kaggle/run_smoke.py main() "
        "under %s" % sys.executable)
    rc = load_module("ber_run_smoke", RUN_SMOKE).main()
    if rc != 0:
        fail("kaggle/run_smoke.py returned %s (output above)" % rc)
    output_dir = locate_smoke_tsvs()
    log("[e2e] step 2/4: assemble team='team' zip from %s" % output_dir)
    zip_path, src_files = assemble("team", output_dir)
    log("[e2e] step 3/4: run the official validator on the tiny outputs")
    run_validator(os.path.join(output_dir, MATCHING),
                  os.path.join(output_dir, CANDIDATE), TEST_TINY, gate=True)
    log("[e2e] step 4/4: assert zip structure")
    check_zip(zip_path, output_dir, src_files)
    log("E2E PASS - %s" % zip_path)
    return 0


def cmd_real(args):
    team = re.sub(r'[<>:"/\\|?*\s]+', "_", args.team.strip())
    if not team:
        fail("--team must be a non-empty name")
    return_zip = os.path.abspath(args.zip)
    if not os.path.isfile(return_zip):
        fail("kaggle_return.zip not found: " + return_zip)
    staging = tempfile.mkdtemp(prefix="submit_staging_")
    try:
        log("[real] step 1/3: read TSVs from %s" % return_zip)
        with zipfile.ZipFile(return_zip) as zf:
            names = zf.namelist()
            for arc in (RETURN_ZIP_MATCHING, RETURN_ZIP_CANDIDATE):
                if arc not in names:
                    fail("%s not in %s; zip entries: %s" % (arc, return_zip, names))
                zf.extract(arc, staging)
                log("[real] extracted %s (%d bytes)"
                    % (arc, os.path.getsize(os.path.join(staging, arc))))
        output_dir = os.path.join(staging, "output")
        log("[real] step 2/3: assemble team=%r zip (plus official validator)"
            % team)
        zip_path, src_files = assemble(team, output_dir)
        run_validator(os.path.join(output_dir, MATCHING),
                      os.path.join(output_dir, CANDIDATE), TEST_REAL, gate=False)
        log("[real] step 3/3: assert zip structure")
        check_zip(zip_path, output_dir, src_files)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log("REAL PASS - submit %s" % zip_path)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Package the ML Challenge 2026 submission zip.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("e2e", help="run kaggle/run_smoke.py, package tiny outputs, "
                               "validate and assert structure")
    rp = sub.add_parser("real", help="package the real downloaded kaggle_return.zip")
    rp.add_argument("--zip", required=True, help="path to kaggle_return.zip "
                                                 "downloaded from the Kaggle run")
    rp.add_argument("--team", required=True, help="team name -> <team>_submission.zip")
    args = ap.parse_args()
    return cmd_e2e(args) if args.cmd == "e2e" else cmd_real(args)


if __name__ == "__main__":
    sys.exit(main())
