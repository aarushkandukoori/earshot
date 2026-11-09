import json, sys, urllib.request
def tree(repo, path="", rev="main"):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{path}?recursive=true&expand=false"
    out, cursor = [], url
    while cursor:
        req = urllib.request.Request(cursor, headers={"User-Agent": "research"})
        with urllib.request.urlopen(req, timeout=60) as r:
            out += json.load(r)
            link = r.headers.get("Link")
        cursor = None
        if link and 'rel="next"' in link:
            cursor = link.split(";")[0].strip("<>")
    return out
for spec in sys.argv[1:]:
    repo, _, path = spec.partition(":")
    try:
        files = [f for f in tree(repo, path) if f.get("type") == "file"]
    except Exception as e:
        print(repo, path, "ERR", e); continue
    by_dir = {}
    for f in files:
        d = f["path"].rsplit("/", 1)[0] if "/" in f["path"] else "."
        n, s = by_dir.get(d, (0, 0)); by_dir[d] = (n + 1, s + f.get("size", 0))
    tot = sum(s for n, s in by_dir.values())
    print(f"== {repo}:{path} files={len(files)} total={tot/1e9:.2f} GB")
    for d, (n, s) in sorted(by_dir.items())[:40]:
        print(f"   {d:70s} n={n:5d} {s/1e9:8.2f} GB  avg={s/max(n,1)/1e6:7.1f} MB")
