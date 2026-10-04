"""fetch_rii_database.py -- download the refractiveindex.info database (CC0, M. N. Polyanskiy,
https://github.com/polyanskiy/refractiveindex.info-database) into ../rii (next to HOPS_NLayer), as used by
screen_rii_database.py, survey_db_nlayer.py and the database scenarios.

    python fetch_rii_database.py                 # catalog + the 1704 pages of the shelves main / other / organic / glass
    python fetch_rii_database.py --dest D:/rii   # elsewhere (then set RII_DB=D:/rii)

Alternative: git clone https://github.com/polyanskiy/refractiveindex.info-database and set RII_DB=<clone>/database.
The pages used by the scenarios themselves are already shipped in HOPS_NLayer/rii_database.
"""
import argparse
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RAW = 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/'
HERE = os.path.dirname(os.path.abspath(__file__))


def get(url, path):
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        open(path, 'wb').write(data)
        return True
    except Exception as e:                           # noqa: BLE001
        print('FAILED', url, e)
        return False


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=os.path.join(HERE, '..', 'rii'))
    ap.add_argument('--shelves', nargs='*', default=['main', 'other', 'organic', 'glass'])
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    get(RAW + 'catalog-nk.yml', os.path.join(dest, 'catalog-nk.yml'))
    cat = yaml.safe_load(open(os.path.join(dest, 'catalog-nk.yml'), encoding='utf-8'))
    paths = []

    def walk(node, shelf=None):
        for it in node:
            if 'SHELF' in it:
                walk(it['content'], it['SHELF'])
            elif 'BOOK' in it:
                walk(it['content'], shelf)
            elif 'PAGE' in it and it.get('data') and shelf in a.shelves:
                paths.append(it['data'])
    walk(cat)
    paths = sorted(set(paths))
    print(f'{len(paths)} pages -> {dest}')
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda p: get(RAW + 'data/' + urllib.parse.quote(p), os.path.join(dest, 'data', p)), paths))
    print(f'{sum(ok)} / {len(paths)} downloaded.  set RII_DB={dest} (or keep it at ../rii, found automatically)')


if __name__ == '__main__':
    main()
