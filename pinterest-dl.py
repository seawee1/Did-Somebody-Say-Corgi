#! /usr/bin/env python3
import signal
import sys

import argparse
import logging

from pinterestDL.pinterest_downloader import PinterestDownloader


if sys.version_info < (3, 6):
    raise RuntimeError("Please call this script with python 3.6 or newer!")


def handle_sig_int(signal, frame):
    """
    Exit gracefully on CTRL+C or other source of SIGINT.

    :param signal: The signal that was received.
    :param frame: The stack frame in which the signal was received.
    :return None.
    """
    logging.warning("Aborted, download may be incomplete.")
    sys.exit(0)


def parse_cmd():
    """
    Parses command line flags that control how pinterest will be scraped.
    Start the script with the '-h' option to read about all the arguments.

    :returns a namespace populated with the arguments supplied (or default arguments, if given).
    """
    parser = argparse.ArgumentParser(description="""Download a pinterest board or tag page. When downloading a tag page,
    and no maximal number of downloads is provided, stop the script with CTRL+C.""")
    # Required arguments
    parser.add_argument(dest="link", help="Link to the pinterest page you want to download.")
    parser.add_argument(dest="dest_folder",
                        help="""Folder into which the board will be downloaded.
                         Folder with board name is automatically created or found inside this folder, if it already exists.
                         If this folder is named like the page to be downloaded, everything will be directly in this folder.""")
    # Optional arguments
    parser.add_argument("-n", "--name", default=None, required=False, dest="board_name",
                        help="""The name for the downloaded page. If not given, will try to extract board name from pinterest.
                        This will also be the name for the folder in which the images are stored.""")
    parser.add_argument("-c", "--count", default=None, type=int, required=False, dest="num_pins",
                        help="""Download only the first 'num_pins' pins found on the page.
                        If bigger than the number of pins on the board, all pins in the board will be downloaded.
                        The default is to download all pins. If you do not specifiy this option on a tag page, where there are more or less infinite pins,
                        just stop the script with CTRL+C.""")
    parser.add_argument("-j", "--threads", default=4, type=int, required=False, dest="nr_threads",
                        help="Number of threads that download images in parallel. Defaults to 4.")
    parser.add_argument("-r", "--resolution", default="0x0", required=False, dest="min_resolution",
                        help="""Minimal resolution for a download image. Input as 'WIDTHxHEIGHT'.""")
    parser.add_argument("-m", "--mode", default="individual", required=False, choices=["individual", "area"],
                        dest="mode",
                        help="""Pick how the resolution limit is treated:
                             'individual': Both image dimensions must be bigger than the given resolution, i.e x >= WIDTH and y >= HEIGHT.
                             'area': The area of the image must be bigger than the provided resolution, i.e. x*y >= WIDTH * HEIGHT.""")
    parser.add_argument("-s", "--skip-limit", default=float("inf"), type=int, required=False, dest="skip_limit",
                        help="""Abort the download after so many pins have been skipped. A pin is skipped if it was already present in the download folder.
                            This way you can download new pins that have been added after your last download. Defaults to infinite.
                            You should not set this to 1, but rather something like 10,
                            because the page is not scraped exactly in the same order as the pins are added.""")
    parser.add_argument("-t", "--timeout", default=15, type=int, required=False, dest="timeout",
                        help="Set the timeout in seconds after which loading a pinterest board will be aborted, if unsuccessfull. Defaults to 15 seconds.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", dest="verbose", required=False,
                        help="Display more detailed output and progress reports.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sig_int)
    arguments = parse_cmd()
    log_level = logging.INFO
    if arguments.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%I:%M:%S')

    with PinterestDownloader(page_timeout=arguments.timeout,
                             num_threads=arguments.nr_threads,
                             min_resolution=arguments.min_resolution,
                             size_compare_mode=arguments.mode) as dl:
        dl.download_board(board_url=arguments.link, download_folder=arguments.dest_folder,
                          num_pins=arguments.num_pins, board_name=arguments.board_name,
                          skip_tolerance=arguments.skip_limit)
