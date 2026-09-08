"""One-time extraction of native Word (OMML) equations from the original guide.

The v1/v2 guide was produced by a pandoc-style pipeline that turned LaTeX into
native Word math. Rebuilding the document from markdown would normally lose
that math, so the OMML fragments are lifted out verbatim and replayed by
``tools/build_guide_docx.py``.

Usage:
    python tools/extract_equations.py [source.docx]
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "QuantLab_Investment_Strategy_and_Research_Guide(2).docx"
TARGET = ROOT / "tools" / "guide_assets" / "equations.json"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Stable slugs keyed by the flattened math text of each distinct equation.
SLUGS = {
    "R2": "r2",
    "Pt": "price_t",
    "rt=Pt/Pt\u22121\u22121.": "daily_return",
    "Rw=t\u2208w\u200b1+rt\u22121.": "weekly_compounding",
    "RB,w=\u03b1w+\u03b3w\u22a4Fw+uw.": "hedge_regression",
    "rpure,t=rB,t\u2212\u03b3t\u22a4ft.": "pure_return",
    "Lt=min10%\u03c3t\u22121,5,\u2001\u2001ft=Ltrpure,t.": "vol_scaling",
    "i": "stock_i",
    "ri,t=\u03b1i+k\u200b\u03b2i,kfk,t+\u03f5i,t.": "stock_regression",
    "k": "factor_k",
    "zi,k=\u03b2i,k\u2212mean\u03b2\u22c5,kstd\u03b2\u22c5,k.": "zscore",
    "ak": "weight_a_k",
    "dk\u2208{\u22121,1}": "direction_d_k",
    "scorei=k\u200bakdkzi,kk\u200bak.": "composite_score",
    "N": "top_n",
    "B": "beta_matrix",
    "w": "weight_vector",
    "\u03b2p=B\u22a4w.": "portfolio_beta",
    "\u03a9": "factor_cov",
    "\u03c3factor2=\u03b2p\u22a4\u03a9\u03b2p.": "factor_variance",
    "\u03c3specific2=i\u200bwi\u03c3\u03f5,i2.": "specific_variance",
    "\u03a3=B\u03a9B\u22a4+D.": "total_covariance",
    "D": "specific_matrix",
    "\u03bb/2": "lambda_half",
    "B\u03a9B\u22a4+D": "factor_risk_model",
    "rf+k\u200b\u03b2ik\u03bbk": "factor_implied_return",
    "\u03bci\u2212rf=k\u03c3i": "equal_sharpe",
    "\u03bc": "mu",
}


def flatten_math(xml: str) -> str:
    return "".join(re.findall(r"<m:t[^>]*>([^<]*)</m:t>", xml))


def namespaced(fragment: str) -> str:
    """Add the namespace declarations the fragment inherited from its parent."""
    return fragment.replace(
        "<m:oMath>",
        f'<m:oMath xmlns:m="{M_NS}" xmlns:w="{W_NS}">',
        1,
    )


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.exists():
        print(f"source document not found: {source}")
        return 1

    with zipfile.ZipFile(source) as archive:
        document = archive.read("word/document.xml").decode("utf-8")

    equations: dict[str, str] = {}
    unmapped: list[str] = []
    for fragment in re.findall(r"<m:oMath>.*?</m:oMath>", document, re.S):
        text = flatten_math(fragment)
        slug = SLUGS.get(text)
        if slug is None:
            unmapped.append(text)
            continue
        equations.setdefault(slug, namespaced(fragment))

    if unmapped:
        print("unmapped equations found; add slugs for:")
        for text in sorted(set(unmapped)):
            print(f"  {text!r}")
        return 1

    missing = sorted(set(SLUGS.values()) - set(equations))
    if missing:
        print(f"slugs never matched: {missing}")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(
        json.dumps(equations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(equations)} equations to {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
