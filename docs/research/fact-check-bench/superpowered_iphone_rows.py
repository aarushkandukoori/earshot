"""Read the Superpowered crowd latency table and print the iPhone rows.

Usage:
    curl -sSL -o data.html https://superpowered.com/latencydata/data.html
    python3 -I superpowered_iphone_rows.py data.html

Why the "generic iPhone" rows matter:
The Superpowered app names a phone by looking up its machine id (for example
"iPhone17,1") in iOS/machines.h. That list stops at iPhone15,3 (iPhone 14 Pro Max),
so iPhone 15 and iPhone 16 phones fall back to the plain name "iPhone".
Every older phone that can run iOS 17 is in the list, so a plain "iPhone" row
on iOS 17 or later must come from an iPhone 15 or iPhone 16 family phone.
"""
import html
import re
import sys

LABELED = [
    "iPhone SE (3rd gen)", "iPhone 12", "iPhone 12 mini", "iPhone 12 Pro Max",
    "iPhone 13", "iPhone 13 Pro", "iPhone 13 Pro Max",
    "iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max",
]


def clean(cell):
    return html.unescape(re.sub(r"<.*?>", "", cell)).replace("\xa0", " ").strip()


def main(path):
    page = open(path, encoding="utf-8", errors="replace").read()
    header = re.search(r"<h3[^>]*>(.*?)</h3>", page, flags=re.S)
    if header:
        print("Table header:", clean(header.group(1)))

    group = None
    rows = []  # (model, os_version, buffer, sample_rate, latency)
    for row in re.findall(r"<tr>(.*?)</tr>", page, flags=re.S):
        head = re.match(r'\s*<td colspan="4">(.*?)</td>\s*$', row, flags=re.S)
        if head:
            group = clean(head.group(1))
            continue
        cells = re.findall(r"<td([^>]*)>(.*?)</td>", row, flags=re.S)
        if len(cells) != 4 or group is None:
            continue
        first_class, first = cells[0]
        rate = re.search(r'title="(\d+) Hz"', cells[2][1])
        rate = rate.group(1) if rate else ""
        if 'class="v1"' in first_class:
            # A one-row model: "Model   version" in the first cell.
            name, _, version = clean(first).rpartition("   ")
            rows.append((name.strip(), version.strip(), clean(cells[2][1]), rate, clean(cells[3][1])))
        else:
            rows.append((group, clean(first), clean(cells[2][1]), rate, clean(cells[3][1])))

    print("\nLabeled iPhone rows (model, iOS, buffer, Hz, round trip):")
    for model in LABELED:
        for r in rows:
            if r[0] == model:
                print("  ", r)

    print("\nPlain 'iPhone' rows on iOS 17 or later (must be iPhone 15/16 family):")
    for r in rows:
        if r[0] == "iPhone":
            try:
                major = int(r[1].split(".")[0])
            except ValueError:
                continue
            if major >= 17:
                print("  ", r)

    named_15_16 = [r for r in rows if r[0].startswith(("iPhone 15", "iPhone 16", "iPhone 17"))]
    print("\nRows labeled iPhone 15/16/17:", len(named_15_16))


if __name__ == "__main__":
    main(sys.argv[1])
