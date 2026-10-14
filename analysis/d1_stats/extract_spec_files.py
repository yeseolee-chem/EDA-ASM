# extract_spec_files.py  —  python extract_spec_files.py SPEC_D1_ALL.md
import sys, re, hashlib, pathlib
FENCE = "`" * 3
text = open(sys.argv[1], encoding="utf-8").read().split("\n")
i, n_ok = 0, 0
while i < len(text):
    m = re.match(r"<!-- FILE: (\S+) sha256=([0-9a-f]{64}) -->", text[i])
    if m:
        path, want = m.group(1), m.group(2)
        j = i + 1
        assert text[j].startswith(FENCE), f"no fence after {path}"
        k = j + 1
        while not text[k].startswith(FENCE):
            k += 1
        body = "\n".join(text[j + 1:k]) + "\n"
        got = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if got != want:
            sys.exit(f"SHA MISMATCH {path}: {got}")
        p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body, encoding="utf-8")
        n_ok += 1; i = k
    i += 1
print("files written:", n_ok)
