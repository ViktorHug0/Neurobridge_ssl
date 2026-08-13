"""Convert answers.tex into what OpenReview's Markdown+MathJax response box actually renders.

Two independent things break a naive paste, and the fix differs for each.

Structure. MathJax implements only math-mode TeX, so \\textbf, \\emph, \\begin{itemize} and
\\item print literally. These have to become Markdown.

Math. Markdown runs before MathJax and consumes characters the TeX still needs:
  * a backslash before punctuation is a Markdown escape, so \\{ reaches MathJax as a bare {
    (a backslash before a letter, as in \\mathbf, is left alone);
  * a bare _ is an emphasis delimiter, so the underscores in a paragraph pair up across the
    $...$ boundaries and are deleted, which loses the subscript and unbalances the delimiters,
    merging several formulas into one blob.
Doubling every backslash and writing each underscore as \\_ survives Markdown exactly: it
turns \\\\ back into \\ and \\_ back into _, so MathJax receives the original TeX.

Spans that are only digits and punctuation are emitted as plain text, since $80$ and 80 render
identically and the markup only adds noise.

Usage:  python tex_to_markdown.py answers.tex -o answers_markdown.md
"""
import argparse
import re

PLAIN = re.compile(r"^[\d\s+\-.,=%()]+$")  # nothing MathJax would render differently


def escape_math(body):
    """Make a TeX span survive Markdown so MathJax sees it unchanged."""
    body = body.strip()
    body = body.replace("\\", "\\\\")   # Markdown eats one level of backslash
    body = body.replace("_", "\\_")     # and would otherwise read _ as emphasis
    return body


def inline_math(body):
    if PLAIN.match(body.strip()):
        return body.strip()
    return f"${escape_math(body)}$"


def cell_to_math(cell):
    """One tabular cell as math-mode content for an array environment."""
    cell = re.sub(r"\$([^$]*)\$", r"\1", cell.strip())   # already inside math
    if not cell:
        return ""
    # a bare variable, possibly with a value, stays italic maths: K, N, K=25
    if re.fullmatch(r"[A-Za-z](\s*=\s*[\d.]+)?", cell):
        return cell
    # anything with prose in it has to be upright, or maths mode sets it as a product of letters
    if re.search(r"[A-Za-z]", re.sub(r"\\[a-zA-Z]+", "", cell)):
        return r"\text{" + cell + "}"
    return cell.replace(" ", r"\,")                      # maths ignores ordinary spaces


def convert_tabular(block, preamble):
    """LaTeX tabular -> MathJax array.

    MathJax accepts the array environment but only l, r, c, | and : in its preamble, and it
    does not implement \\multicolumn, so a spanning label is placed in the first column of its
    group and the remaining cells are left empty.
    """
    preamble = "".join(c for c in preamble if c in "lrc|:") or "c"
    body = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(("\\begin", "\\end")):
            continue
        if line.startswith("\\hline"):
            body.append("\\hline")
            line = line[len("\\hline"):].strip()
            if not line:
                continue
        line = re.sub(r"\\\\\s*$", "", line)
        cells = []
        for cell in line.split("&"):
            cell = cell.strip()
            span = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", cell)
            if span:
                cells.append(cell_to_math(span.group(2)))
                cells += [""] * (int(span.group(1)) - 1)
            else:
                cells.append(cell_to_math(cell))
        body.append(" & ".join(cells) + " \\\\")
    if not body:
        return ""
    inner = "\n".join(body)
    return "$$\n" + escape_math(f"\\begin{{array}}{{{preamble}}}\n{inner}\n\\end{{array}}") + "\n$$"


def convert(text):
    text = text.split("\\begin{document}", 1)[-1].split("\\end{document}", 1)[0]
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    text = re.sub(r"\\maketitle", "", text)
    text = text.replace("---", "\u2014")            # before tables, or it eats the |---| rule

    text = re.sub(r"\\begin\{center\}\s*(.*?)\s*\\end\{center\}", lambda m: m.group(1), text, flags=re.S)
    text = re.sub(r"\\small|\\setlength\{\\tabcolsep\}\{[^}]*\}", "", text)
    text = re.sub(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                  lambda m: "\n\n" + convert_tabular(m.group(2), m.group(1)) + "\n", text, flags=re.S)

    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n\n$$" + escape_math(m.group(1)) + "$$\n", text, flags=re.S)
    text = re.sub(r"\$([^$]*)\$", lambda m: inline_math(m.group(1)), text)

    text = re.sub(r"\\section\{([^}]*)\}", r"\n\n## \1\n", text)
    text = re.sub(r"\\subsection\{([^}]*)\}", r"\n\n### \1\n", text)
    text = re.sub(r"\\answer\{", "\n**Answer:** ", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = re.sub(r"\\todo", "**(TODO)**", text)
    text = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", text)
    text = re.sub(r"(?m)^\s*\\item\s*", "- ", text)

    text = re.sub(r"(?<=\w)--(?=\w)", "\u2013", text)
    text = re.sub(r"~?\\ref\{([^}]*)\}", r" \1", text)
    text = re.sub(r"~?\\cite\{([^}]*)\}", r" [\1]", text)
    # \% and \& are LaTeX escapes only in prose. Inside a maths span they must stay escaped:
    # an unescaped % starts a TeX comment and would swallow the rest of the array row.
    def unescape_prose(chunk):
        return chunk.replace("\\%", "%").replace("\\&", "&").replace("~", " ")

    parts = re.split(r"(\$\$.*?\$\$|\$[^$]*\$)", text, flags=re.S)
    text = "".join(p if i % 2 else unescape_prose(p) for i, p in enumerate(parts))
    text = re.sub(r"^\s*\}\s*$", "", text, flags=re.M)   # closing brace of \answer{...}
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    out = convert(open(args.source).read())
    open(args.output, "w").write(out)

    outside = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", "", out, flags=re.S)  # ignore what is inside math
    leftover = sorted(set(re.findall(r"\\[a-zA-Z]+", outside)))
    spans = re.findall(r"\$\$.*?\$\$|\$[^$]*\$", out, flags=re.S)
    bad = [s[:60] for s in spans if re.search(r"(?<!\\)\\[a-zA-Z]", s) or re.search(r"(?<!\\)_", s)]
    print(f"wrote {args.output} ({len(out)} chars, {len(spans)} math spans)")
    print("text-mode LaTeX left outside math:", leftover or "none")
    print("math spans with an unescaped backslash or underscore:", bad or "none")


if __name__ == "__main__":
    main()
