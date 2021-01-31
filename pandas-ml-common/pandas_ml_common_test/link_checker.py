import os
import re
import tarfile
import requests
from unittest import TestCase

import markdown
from lxml import etree
__version__ = '0.2.0'

url = 'https://github.com/KIC/pandas-ml-quant/pandas-ml-common'


def fix_github_links(line):
    fixed_images = re.sub(r'(^\[ghi\d+]:\s+)', f'\\1{url}/raw/{__version__}/', line)
    fixed_location = re.sub(r'(^\[ghl\d+]:\s+)', f'\\1{url}/tree/{__version__}/', fixed_images)
    fixed_files = re.sub(r'(^\[ghf\d+]:\s+)', f'\\1{url}/blob/{__version__}/', fixed_location)
    return fixed_files


def check_links_in_dist(dist_file, file):
    file = os.path.join(os.path.basename(dist_file.replace('.tar.gz', '')), file)
    with tarfile.open(dist_file, mode='r:gz') as tar:
        md = tar.extractfile(file).readlines()
        md = '\n'.join([fix_github_links(l.decode('utf-8')) for l in md])

        doc = etree.fromstring('<html>' + markdown.markdown(md) + '</html>')
        for link in doc.xpath('//a'):
            print(link.text, link.get('href'))
            print(requests.get(link.get('href')).status_code)
        for image in doc.xpath('//img'):
            print(image.text, image.get('src'))
            print(requests.get(image.get('src')).status_code)


class TestLinkChecker(TestCase):

    def test_link_checker(self):
        check_links_in_dist('/tmp/foo.tgz/pandas-ml-utils-0.2.1.tar.gz', 'Readme.md')
