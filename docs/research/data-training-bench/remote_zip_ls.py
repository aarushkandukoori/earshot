"""List members of a remote zip via HTTP range requests only (reads the central directory, not the data)."""
import sys, zipfile, collections, fsspec
for url in sys.argv[1:]:
    fs = fsspec.filesystem("http")
    with fs.open(url, "rb", block_size=1 << 20) as fh:
        z = zipfile.ZipFile(fh)
        infos = z.infolist()
        by = collections.defaultdict(lambda: [0, 0])
        for i in infos:
            parts = i.filename.strip("/").split("/")
            key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
            by[key][0] += 1; by[key][1] += i.file_size
        print(f"== {url.split('/')[-1][:60]} members={len(infos)} uncompressed={sum(i.file_size for i in infos)/1e9:.2f} GB")
        for k, (n, s) in sorted(by.items())[:25]:
            print(f"   {k[:80]:80s} n={n:6d} {s/1e9:7.3f} GB")
