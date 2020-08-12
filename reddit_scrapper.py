from datetime import datetime, timedelta
from pandas import json_normalize
import pandas
from os.path import join
from pathlib import Path
import math

#import requests
import aiohttp
import asyncio
import aiohttp_retry
from urllib import parse

# Async get_submissions
async def get_submissions(**kwargs):
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.pushshift.io/reddit/submission/search/', params=kwargs) as resp:
            response = await resp.json()
            return response['data']

# Sync get_submission
#def get_submissions(**kwargs):
#    r = requests.get('https://api.pushshift.io/reddit/submission/search/', params=kwargs)
#    try:
#        data = r.json()['data']
#        return data
#    except:
#        return None

# Chunks comment_ids
def comment_ids_chunks(comment_ids):
    max_len = 8190 # max length of GET url
    base_len = len('https://api.pushshift.io/reddit/comment/search/') + len(parse.urlencode(comments_params)) # Fixed for every request
    ids_max_len = max_len - base_len # Max length of url part containing comment ids
    id_len = 12 # Looks like this: &ids=drm7806
    max_ids_per_request = ids_max_len // id_len # Resulting max amount of comment_ids per request
    chunks = math.ceil(len(comment_ids) / max_ids_per_request) # Resulting amount of necessary requests

    # Generator yielding comment_ids
    for i in range(chunks):
        start = i * max_ids_per_request
        end = start + max_ids_per_request
        yield comment_ids[start:end]


async def download_submission(submission, delay):
    # Delay based on task_id to reduce 429 status (too many requests)
    await asyncio.sleep(delay)

    # We use aiohttp_retry to maximize response rate
    #async with aiohttp.ClientSession() as session:
    async with aiohttp_retry.RetryClient(raise_for_status = False) as session:

        # Get comment ids of submission
        comment_ids = None
        async with session.get('https://api.pushshift.io/reddit/submission/comment_ids/' + str(submission['id']), **retry_params) as resp:
            response = await resp.json()
            comment_ids = response['data']

        # Get the actual comments
        comments = []
        # Comment request has to be chunked, because request must not be longer than 8190 characters
        for comment_ids_chunk in comment_ids_chunks(comment_ids):
            async with session.get('https://api.pushshift.io/reddit/comment/search/', params = {**comments_params, 'ids':comment_ids_chunk}, **retry_params) as resp:
                response = await resp.json()
                comments.extend(response['data'])

        # Prepare image url
        image_url = submission['url']
        extension = ''
        if '.png' in image_url:
            extension = '.png'
        elif '.jpg' in image_url or '.jpeg' in image_url:
            extension = '.jpeg'
        elif 'imgur' in image_url and not '.gifv' in image_url:
            if not '.jpg' in image_url:
                image_url += '.jpg'
            if not 'i.imgur' in image_url:
                image_url = image_url.replace('imgur', 'i.imgur')
            if 'm.i.imgur' in image_url:
                image_url = image_url.replace('m.i.imgur', 'i.imgur')
            extension = '.jpeg'
        else:
            return

        # Prepare comments dataframe
        for c in comments:
            if 't1_' in c['parent_id']:
                c['is_child'] = True
            else:
                c['is_child'] = False
            c['parent_id'] = c['parent_id'].split('_')[1]
            c['body'] = c['body'].replace('\n', ' ')
        df = json_normalize(comments)

        # Download image and save
        async with session.get(image_url) as resp:
            with open(join(images_dir, submission['id'] + extension), mode = 'wb') as f:
                f.write(await resp.read())

        # Save comments
        with open(join(comments_dir, '{}.csv'.format(submission['id'])), mode='w', encoding='utf-8') as f:
            df.to_csv(f, index=False, line_terminator='\n')

        return submission


async def main():
    global before

    scrapped = 0
    while scrapped < submission_to_scrap:
        print('Scrapped {:d}/{:d} submissions...'.format(scrapped, submission_to_scrap))

        # This actually doesn't have to be asynchronous, as it is a single GET request
        done, _ = await asyncio.wait([get_submissions(before=before, **submissions_params)])
        submissions = list(done)[0].result()

        # Create tasks
        tasks = []
        for i, s in enumerate(submissions):
            tasks.append(download_submission(s, i))
            before = s['created_utc']

        # Wait until finished
        done = await asyncio.gather(*tasks, return_exceptions=True)
        for d in done:
            if isinstance(d, Exception):
                print(d)

        # Unsuccesful ones are of type Exception, as return_exceptions=True for asyncio.gather
        successful = [x for x in done if isinstance(x, dict)]

        # Append to submission index
        df = json_normalize(successful)
        with open(join(output_dir, 'submissions.csv'), 'a', encoding='utf-8') as f:
            df.to_csv(f, index = False, header=f.tell()==0, line_terminator='\n')

        scrapped += len(successful)

if __name__ == '__main__':
    submission_to_scrap = 1000000

    submissions_params = dict(
        subreddit='NatureIsFuckingLit',
        filter=['id', 'title', 'author', 'score', 'created_utc', 'num_comments', 'is_video', 'url', 'permalink'],
        num_comments='>10',
        is_video='false',
        score='>100',
        size=20
    )
    comments_params = dict(
        filter=['id', 'author', 'score', 'body', 'parent_id', 'permalink']
    )

    retry_params = dict(
        retry_attempts = 5,
        retry_start_timeout = 5.0,
        retry_max_timeout = 15.0,
        retry_factor = 2.0,
        retry_for_statuses = {429}
    )

    database = 'database_1597063213' # Set to None to start from today, else continues on specified database
    if database is None:
        # Save directory
        # output_dir = 'database_{:d}'.format(before)
        before = int((datetime.now() - timedelta(days=1)).timestamp())
        output_dir = 'H:\\reddit\\' + 'database_{:d}'.format(before)
    else:
        output_dir = 'H:\\reddit\\' + database
        df = pandas.read_csv(join(output_dir, "submissions.csv"))
        before = int(df['created_utc'].min())
    comments_dir = join(output_dir, 'comments')
    images_dir = join(output_dir, 'images')
    comments_dir = join(output_dir, 'comments')
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(comments_dir).mkdir(parents=True, exist_ok=True)

    asyncio.run(main())