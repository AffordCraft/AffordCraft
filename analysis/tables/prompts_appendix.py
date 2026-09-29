"""Appendix prompt listing: extracts the three multimodal-model prompts verbatim from the frozen method code (--code: this
repository, i.e. affordcraft/vision.py ground() and rank() and scripts/detect_scene_objects.py, or the frozen code archive
with method/affordcraft/vision.py and tools/detect_scene_objects.py) and writes <out>/sections/appendix_prompts.tex. The
prompt expressions are parsed with ast: string constants are rendered verbatim (LaTeX-escaped, in a small typewriter
block), and every non-constant operand (the per-input fields the code appends) is rendered as an angle-bracket
placeholder named below. The detection prompt head is asserted equal to the prompt_head recorded in the frozen detection
definition of the cluttered-image experiment (prompts/detection_prompt_head.json), so the listing is tied to the recorded run.
"""

import ast, io, json, tarfile
from pathlib import Path
import argparse

_AP = argparse.ArgumentParser(description="appendix prompt listing")
_AP.add_argument(
    "--inputs",
    default=str(Path(__file__).resolve().parents[1] / "results"),
    help="recorded inputs (layout of analysis/results)",
)
_AP.add_argument(
    "--out",
    default=str(Path(__file__).resolve().parents[1] / "out"),
    help="output directory (tables/, figures/, facts/, sections/)",
)
_AP.add_argument(
    "--code",
    default=str(Path(__file__).resolve().parents[2]),
    help="repository root, or the frozen code archive (.tar.gz)",
)
_A = _AP.parse_args()
IN = Path(_A.inputs)
OUT = Path(_A.out)
TAR = Path(_A.code)
REPO_MEMBER = {
    "method/affordcraft/vision.py": "affordcraft/vision.py",
    "tools/detect_scene_objects.py": "scripts/detect_scene_objects.py",
}
PLACEHOLDER = {
    "case['instruction']": "⟨task instruction⟩",
    "json.dumps({k: grounding.get(k) for k in ('operated_part', 'motion_family')})": "⟨grounded operated part and motion family, JSON⟩",
    "json.dumps(meta)": "⟨candidate list: id, category, motion types, semantic parts, joint types; JSON⟩",
    "json.dumps(categories)": "⟨catalog category vocabulary, JSON list⟩",
}


def source(member):
    if TAR.is_dir():
        return (TAR / REPO_MEMBER[member]).read_text(encoding="utf-8")  # repository layout
    with tarfile.open(TAR) as t:
        return t.extractfile(member).read().decode("utf-8")


def prompt_expr(src, func=None):
    """the value expression of the assignment `prompt=(...)` inside `func` (or at module level of the given function-less script)"""
    tree = ast.parse(src)
    scope = tree
    if func:
        scope = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func]
        assert len(scope) == 1, func
        scope = scope[0]
    hits = [
        n
        for n in ast.walk(scope)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "prompt"
    ]
    assert len(hits) == 1, (func, len(hits))
    return hits[0].value


def render(node):
    """list of ('text',s) / ('field',name) parts of a + chain of string constants and expressions"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return render(node.left) + render(node.right)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [("text", node.value)]
    key = ast.unparse(node)
    assert key in PLACEHOLDER, key
    return [("field", PLACEHOLDER[key])]


def esc(s):
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("&", r"\&"),
        ("$", r"\$"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(a, b)
    return s


def tex(parts):
    out = []
    for kind, s in parts:
        if kind == "text":
            s = esc(s).replace(
                "\n", r"\\" + "\n"
            )  # the rank prompt's embedded newlines (before the grounded-part and candidate fields) become line breaks
            out.append(s)
        else:
            out.append(r"$\langle$" + esc(s.strip("⟨⟩")) + r"$\rangle$")
    return "".join(out)


def main():
    vision = source("method/affordcraft/vision.py")
    detect = source("tools/detect_scene_objects.py")
    prompts = [
        ("Grounding (single-object inputs).", "ground", prompt_expr(vision, "ground"), "the full photograph"),
        (
            "Candidate ranking.",
            "rank",
            prompt_expr(vision, "rank"),
            "the photograph, the crop of the grounded region, and one preview per candidate",
        ),
        ("Detection (cluttered images).", "detect", prompt_expr(detect), "the full cluttered image"),
    ]
    freeze = json.load(open(IN / "prompts/detection_prompt_head.json", encoding="utf-8"))
    det_text = "".join(s for k, s in render(prompts[2][2]) if k == "text")
    assert det_text == freeze["prompt_head"], "detection prompt differs from the frozen prompt head"
    L = [
        "% Generated by analysis/tables/prompts_appendix.py from the frozen method code. Edit the generator, not this file."
    ]
    for title, name, expr, images in prompts:
        parts = render(expr)
        L.append(r"\paragraph{" + title + "}")
        L.append("Images: " + images + ".")
        L.append(r"\begin{quote}\small\ttfamily\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000")
        L.append(tex(parts))
        L.append(r"\end{quote}")
    (OUT / "sections").mkdir(parents=True, exist_ok=True)
    (OUT / "sections/appendix_prompts.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    rec = {
        name: {
            "parts": render(expr),
            "source_member": "method/affordcraft/vision.py" if name != "detect" else "tools/detect_scene_objects.py",
        }
        for _, name, expr, _ in prompts
    }
    (OUT / "facts").mkdir(parents=True, exist_ok=True)
    (OUT / "facts/prompts.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print("prompts written:", [(n, sum(len(s) for k, s in rec[n]["parts"] if k == "text")) for n in rec])


if __name__ == "__main__":
    main()
