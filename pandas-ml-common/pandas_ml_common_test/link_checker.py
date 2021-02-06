import os
import sys
import tarfile
import zipfile
from unittest import TestCase

import markdown
import requests
from lxml import etree

from pandas_ml_common_test.config import DATA_PATH


def check_links_in_dist(dist_file, file):
    if dist_file.endswith(".tar.gz"):
        file = os.path.join(os.path.basename(dist_file.replace('.tar.gz', '')), file)
        with tarfile.open(dist_file, mode='r:gz') as tar:
            md = tar.extractfile(file).read().decode('utf-8')
            check_links_in_markdown(md)
    elif dist_file.endswith(".zip"):
        file = os.path.join(os.path.basename(dist_file.replace('.zip', '')), file)
        with zipfile.ZipFile(dist_file, 'r') as zip:
            md = zip.read(file).decode('utf-8')
            check_links_in_markdown(md)
    else:
        raise ValueError("unknown dist packaging, allowed .tar.gz|.zip")


def check_links_in_markdown(md):
    if len(md) <= 0:
        raise ValueError("Empty Readme!")

    doc = etree.fromstring('<html>' + markdown.markdown(md) + '</html>')
    for link in doc.xpath('//a'):
        print(link.text, link.get('href'))
        assert requests.get(link.get('href')).status_code == 200, f"failed url {link.get('href')}"
    for image in doc.xpath('//img'):
        print(image.text, image.get('src'))
        assert requests.get(image.get('src')).status_code == 200, f"failed url {image.get('src')}"

