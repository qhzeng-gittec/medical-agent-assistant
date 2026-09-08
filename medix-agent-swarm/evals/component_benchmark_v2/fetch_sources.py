"""Download the official MedlinePlus health-topic XML snapshot used by this benchmark."""
import argparse
from hashlib import sha256
import io
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from api import write

URL = 'https://medlineplus.gov/xml/mplus_topics_compressed_2026-09-05.zip'


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'medlineplus_20260905.zip'
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            path.write_bytes(response.read())
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        tree = ET.fromstring(archive.read('mplus_topics_2026-09-05.xml'))
    topics = [{'id':t.get('id'), 'title':t.get('title'), 'url':t.get('url'),
               'summary_html':t.findtext('full-summary'), 'groups':[g.text for g in t.findall('group')]}
              for t in tree.findall('health-topic') if t.get('language') == 'English']
    write(output/'topics.json', topics)
    write(output/'download.json', {'url':URL,'sha256':sha256(path.read_bytes()).hexdigest(),'english_topics':len(topics),
                                   'attribution':'Source: MedlinePlus, National Library of Medicine'})
    print(f'Downloaded {len(topics)} English health-topic summaries.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
