import sys

from pandas_ml_common_test.link_checker import check_links_in_dist


if __name__ == "__main__":
    _, dist = sys.argv
    check_links_in_dist(dist, 'Readme.md')