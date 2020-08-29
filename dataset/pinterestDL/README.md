# PinterestDL: Easily download Pinterest boards and tag pages

This python script allows you to download pinterest pages in a fast way, filtered by size of the images, and many more options.
**Warning**: Some functionality currently does not behave as expected.
The website scraping should be re-written, but I currently don't have time for this.
Feel free to send a pull request.

## Installation
Tested on Ubuntu 18.04 LTS.

### Requirements
1. Python >= 3.6

### Instructions

- Recommended is a new python3.6 environment:

  ```
  python3 -m venv ./pinDL_venv
  source ./pinDL_venv/bin/activate
  ```

- Clone the git repository and switch to it:

  ```
  git clone https://github.com/RunOrVeith/pinterestDL
  cd pinterestDL
  ```

- Install the python requirements:

    ```
    pip3 install -r requirements.txt
    ```

- Install geckodriver (The geckodriver needs to be found in your path, if you already have it you can skip this).
  This script will install the latest releast to `/usr/bin/geckodriver`:

   ```
   $SHELL /scripts/download_gecko_driver_linux.sh
   ```

- Optional: Create a symlink to the download script (if you don't do this, just replace the script name with the full path to the script):

  ```
  sudo ln -s `pwd`/pinterestDL/pinterest-dl.py /usr/local/bin/pinterest-dl
  ```

## Usage

The package pinterestDL contains a command line tool, with which you can download pins from pinterest pages, i.e. boards or search results (tag pages).

1. Go to the page you want to download in your favorite browser and copy the link.
2. Call the script (the quotes are important, if you do not put them the shell will interpret the link as a path)

  ```pinterest-dl "paste your link here" $HOME/Pictures```

**Warning**: Currently the option to automatically extract the number of pins in a board is broken.
Specify the number of pins to download with `-c` or just stop the script.
There will be more pins downloaded than the board is big.

### Command line options

Call ```pinterest-dl -h``` to see these instructions.

    usage: pinterest-dl [-h] [-n BOARD_NAME] [-c NUM_PINS] [-j NR_THREADS]
                        [-r MIN_RESOLUTION] [-m {individual,area}] [-s SKIP_LIMIT]
                        [-t TIMEOUT] [-v]
                        link dest_folder

    Download a pinterest board or tag page. When downloading a tag page, and no
    maximal number of downloads is provided, stop the script with CTRL+C.

    positional arguments:
      link                  Link to the pinterest page you want to download.
      dest_folder           Folder into which the board will be downloaded. Folder
                            with board name is automatically created or found
                            inside this folder, if it already exists. If this
                            folder is named like the page to be downloaded,
                            everything will be directly in this folder.

    optional arguments:
      -h, --help            show this help message and exit
      -n BOARD_NAME, --name BOARD_NAME
                            The name for the downloaded page. If not given, will
                            try to extract board name from pinterest. This will
                            also be the name for the folder in which the images
                            are stored.
      -c NUM_PINS, --count NUM_PINS
                            Download only the first 'num_pins' pins found on the
                            page. If bigger than the number of pins on the board,
                            all pins in the board will be downloaded. The default
                            is to download all pins. If you do not specifiy this
                            option on a tag page, where there are more or less
                            infinite pins, just stop the script with CTRL+C.
      -j NR_THREADS, --threads NR_THREADS
                            Number of threads that download images in parallel.
                            Defaults to 4.
      -r MIN_RESOLUTION, --resolution MIN_RESOLUTION
                            Minimal resolution for a download image. Input as
                            'WIDTHxHEIGHT'.
      -m {individual,area}, --mode {individual,area}
                            Pick how the resolution limit is treated:
                            'individual': Both image dimensions must be bigger
                            than the given resolution, i.e x >= WIDTH and y >=
                            HEIGHT. 'area': The area of the image must be bigger
                            than the provided resolution, i.e. x*y >= WIDTH *
                            HEIGHT.
      -s SKIP_LIMIT, --skip-limit SKIP_LIMIT
                            Abort the download after so many pins have been
                            skipped. A pin is skipped if it was already present in
                            the download folder. This way you can download new
                            pins that have been added after your last download.
                            Defaults to infinite. You should not set this to 1,
                            but rather something like 10, because the page is not
                            scraped exactly in the same order as the pins are
                            added.
      -t TIMEOUT, --timeout TIMEOUT
                            Set the timeout in seconds after which loading a
                            pinterest board will be aborted, if unsuccessfull.
                            Defaults to 15 seconds.
      -v, --verbose         Display more detailed output and progress reports.


## Fair Use Information

Please respect the rights of the image right holders that you download. Also read Pinterest's [Terms of Service](https://policy.pinterest.com/en/terms-of-service), especially the [copy-right part](https://policy.pinterest.com/en/copyright).

The creator of this script takes no responsibility for misuse by any user.
