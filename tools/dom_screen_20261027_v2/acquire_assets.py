"""Acquire pinned, small public source/weight assets without executing them."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'external/dom_splendor_426fa50'
COMMIT = '426fa50e06fe99b40620131ca962f418824bb73a'
REPO = 'domdomegg/splendor'
WEIGHT = 'play/weights.bin'
WEIGHT_BLOB = '73153f137bd98c23cb1fb36300efde98757c600a'

def get(url, cap):
    req = urllib.request.Request(url, headers={'User-Agent': 'SplendorResearchQualification'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise ValueError('Asset size cap exceeded')
    return data

def main():
    if DEST.exists():
        raise FileExistsError('Never overwrite an acquired asset directory')
    tree = json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1', 4*1024**2))
    assert not tree.get('truncated')
    chosen = [x for x in tree['tree'] if x['type'] == 'blob' and (
        x['path'] in ('LICENSE', 'README.md', 'Cargo.toml', 'Cargo.lock', WEIGHT) or
        x['path'].endswith(('.rs', '.js', '.ts', '.py', '.json', '.toml')))]
    assert sum(x['size'] for x in chosen) < 256*1024**2
    assert next(x for x in chosen if x['path'] == WEIGHT)['sha'] == WEIGHT_BLOB
    assert next(x for x in chosen if x['path'] == WEIGHT)['size'] == 1757492
    DEST.mkdir(parents=True)
    records = []
    for row in chosen:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{row["path"]}'
        data = get(url, row['size'])
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert len(data) == row['size'] and blob == row['sha'], row['path']
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        records.append(dict(path=row['path'], bytes=len(data), git_blob_sha=blob,
                            sha256=hashlib.sha256(data).hexdigest(), url=url))
    manifest = dict(repository=REPO, commit=COMMIT, files=records,
                    total_bytes=sum(x['bytes'] for x in records), executed=False,
                    weight_git_blob=WEIGHT_BLOB, historical_weights=True)
    with (DEST/'asset_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(files=len(records), bytes=manifest['total_bytes'],
                         weight=next(x for x in records if x['path'] == WEIGHT))))

if __name__ == '__main__':
    main()
