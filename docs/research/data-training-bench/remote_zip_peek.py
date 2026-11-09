import sys, io, zipfile, fsspec, soundfile as sf
url, prefix = sys.argv[1], sys.argv[2]
fs = fsspec.filesystem("http")
with fs.open(url, "rb", block_size=1 << 20) as fh:
    z = zipfile.ZipFile(fh)
    names = [n for n in z.namelist() if n.startswith(prefix) and n.lower().endswith(".wav")]
    for n in names[:3]:
        data = z.read(n)
        info = sf.info(io.BytesIO(data))
        print(n, info.samplerate, info.channels, info.subtype, round(info.duration, 2), "s")
    print("count", len(names))
