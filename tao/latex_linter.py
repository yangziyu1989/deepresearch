"""
LaTeX Linter — catches broken references, orphaned labels, and citation mismatches.

Run before every pdflatex compile to prevent ?? in output.

Usage:
    python -m tao.latex_linter path/to/paper.tex

Exit codes:
    0 — all OK
    1 — errors found (would produce ?? in PDF)
    2 — warnings only (unused labels, uncited bibitems)
"""
import re
import sys
from pathlib import Path
from collections import Counter


def lint_latex(tex_path: str) -> dict:
    """Lint a LaTeX file for broken references and citation issues.

    Returns dict with 'errors' (list of str) and 'warnings' (list of str).
    """
    tex = Path(tex_path).read_text(encoding="utf-8")
    errors = []
    warnings = []

    # ── 1. Labels vs Refs ──────────────────────────────────────────
    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = re.findall(r'\\ref\{([^}]+)\}', tex)
    ref_counts = Counter(refs)

    # Broken refs (ref to non-existent label → ?? in PDF)
    for ref, count in ref_counts.items():
        if ref not in labels:
            # Find line number
            for i, line in enumerate(tex.split('\n'), 1):
                if f'\\ref{{{ref}}}' in line:
                    errors.append(
                        f"BROKEN REF: \\ref{{{ref}}} on line {i} — no matching \\label{{{ref}}} exists"
                    )
                    break

    # Unused labels (defined but never referenced — warning only)
    used_refs = set(refs)
    for label in labels:
        if label not in used_refs:
            warnings.append(f"UNUSED LABEL: \\label{{{label}}} is defined but never referenced")

    # Duplicate labels (would cause wrong ref targets)
    label_list = re.findall(r'\\label\{([^}]+)\}', tex)
    label_counts = Counter(label_list)
    for label, count in label_counts.items():
        if count > 1:
            errors.append(f"DUPLICATE LABEL: \\label{{{label}}} defined {count} times")

    # ── 2. Citations vs Bibitems ───────────────────────────────────
    # Extract all citation keys from \citep{}, \citet{}, \cite{}
    cite_pattern = r'\\(?:citep|citet|cite)\{([^}]+)\}'
    cite_matches = re.findall(cite_pattern, tex)
    cited_keys = set()
    for match in cite_matches:
        for key in match.split(','):
            cited_keys.add(key.strip())

    # Extract all bibitem keys
    bibitem_keys = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', tex))

    # Broken citations (cited but no bibitem → ?? in PDF)
    for key in cited_keys:
        if key not in bibitem_keys:
            for i, line in enumerate(tex.split('\n'), 1):
                if key in line and ('\\cite' in line):
                    errors.append(
                        f"BROKEN CITE: \\cite*{{{key}}} on line {i} — no matching \\bibitem{{{key}}}"
                    )
                    break

    # Uncited bibitems (bibitem exists but never cited — warning)
    for key in bibitem_keys:
        if key not in cited_keys:
            warnings.append(f"UNCITED BIBITEM: \\bibitem{{{key}}} exists but is never cited")

    # ── 3. Equation refs ───────────────────────────────────────────
    eqrefs = re.findall(r'\\eqref\{([^}]+)\}', tex)
    for ref in eqrefs:
        if ref not in labels:
            errors.append(f"BROKEN EQREF: \\eqref{{{ref}}} — no matching \\label")

    # ── 4. Common LaTeX issues ─────────────────────────────────────
    # Table/figure without label (likely to cause broken refs elsewhere)
    envs = re.findall(r'\\begin\{(table|figure)\*?\}(.*?)\\end\{(?:table|figure)\*?\}', tex, re.DOTALL)
    for env_type, env_body in envs:
        if '\\label{' not in env_body and '\\caption{' in env_body:
            # Find line number of this environment
            idx = tex.find(env_body[:50])
            line_num = tex[:idx].count('\n') + 1 if idx >= 0 else '?'
            warnings.append(
                f"UNLABELED {env_type.upper()}: {env_type} with \\caption but no \\label near line {line_num}"
            )

    return {"errors": errors, "warnings": warnings}


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tao.latex_linter path/to/paper.tex")
        sys.exit(2)

    tex_path = sys.argv[1]
    if not Path(tex_path).exists():
        print(f"File not found: {tex_path}")
        sys.exit(2)

    result = lint_latex(tex_path)
    errors = result["errors"]
    warnings = result["warnings"]

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}) — these will produce ?? in PDF:")
        print(f"{'='*60}")
        for e in errors:
            print(f"  ✗ {e}")

    if warnings:
        print(f"\n{'='*60}")
        print(f"WARNINGS ({len(warnings)}):")
        print(f"{'='*60}")
        for w in warnings:
            print(f"  ⚠ {w}")

    if not errors and not warnings:
        print("✓ All references, citations, and labels are consistent.")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(errors)} errors, {len(warnings)} warnings")
    print(f"{'='*60}")

    if errors:
        sys.exit(1)
    elif warnings:
        sys.exit(0)  # warnings don't block
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
