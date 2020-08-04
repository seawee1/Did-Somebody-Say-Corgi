import requests
import time
import json
from datetime import datetime, timedelta
from pandas import json_normalize
import pandas
from os.path import join
from pathlib import Path

def get_comments(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=kwargs)
    try:
        data = r.json()['data']
    except:
        return None
    return data

def get_submissions(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/submission/search/", params=kwargs)
    try:
        data = r.json()['data']
    except:
        return None
    return data

def get_comment_ids_of_submission(submission_id):
    r = requests.get("https://api.pushshift.io/reddit/submission/comment_ids/" + str(submission_id))
    try:
        comment_ids = r.json()['data']
    except:
        return None
    return comment_ids

# Scrapping parameters
submission_to_crawl = 1000000
#days_to_crawl = 7
#after=(datetime.now() - timedelta(days=days_to_crawl)).timestamp()

before=int((datetime.now() - timedelta(days=1)).timestamp())
submissions_params = dict(
    subreddit = 'NatureIsFuckingLit',
    filter = ['id', 'title', 'author', 'score', 'created_utc', 'num_comments', 'is_video', 'url', 'permalink'],
    num_comments = '>10',
    is_video = 'false',
    score = '>100',
    size = 50
)
comments_params = dict(
    filter = ['id', 'author', 'score', 'body', 'parent_id', 'permalink']
)

#filter_submissions = ['id', 'title', 'author', 'score', 'created_utc', 'num_comments', 'is_video', 'url']
#filter_comments = ['id', 'author', 'score', 'body', 'parent_id']

# Save directory
#output_dir = 'database_{:d}'.format(before)
output_dir = 'H:\\reddit\\' + 'database_{:d}'.format(before)
images_dir = join(output_dir, 'images')
comments_dir = join(output_dir, 'comments')
Path(images_dir).mkdir(parents=True, exist_ok=True)
Path(comments_dir).mkdir(parents=True, exist_ok=True)

scrapped = 0

while True:
    print('Scrapped {:d}/{:d} submissions...'.format(scrapped, submission_to_crawl))
    # Save all scrapped submissions in this list
    submissions_lst = []
    submissions = get_submissions(before=before, **submissions_params)
    if not submissions: 
        print('No submissions left!')     
        exit()

    for s in submissions:
        #print(s)

        # Download image
        image_url = s['url']
        if '.png' in image_url:
            extension = '.png'
        elif '.jpg' in image_url or '.jpeg' in image_url:
            extension = '.jpeg'
        elif 'imgur' in image_url and not '.gifv' in image_url:
            if not '.jpg' in image_url:
                image_url = image_url.replace('imgur', 'i.imgur')
                image_url += '.jpg'
            extension = '.jpeg'
            print(image_url)
        else:
            continue

        # TODO: Async
        r = requests.get(image_url)
        if(r.status_code == 200):
            with open(join(images_dir, s['id'] + extension),"bx") as f:
                f.write(r.content)
        else:
            continue

        # Crawl comments
        # TODO: Async
        comment_ids = get_comment_ids_of_submission(s['id'])
        if comment_ids is None:
            continue
        comments = get_comments(ids=comment_ids, **comments_params)
        if comments is None:
            continue
        
        for c in comments:
            if 't1_' in c['parent_id']:
                c['is_child'] = True
            else:
                c['is_child'] = False
            c['parent_id'] = c['parent_id'].split('_')[1]
            c['body'] = c['body'].replace('\n', ' ')
            
        df = json_normalize(comments)
        with open(join(comments_dir, "{}.csv".format(s['id'])), 'w', encoding='utf-8') as f:
            df.to_csv(f, index = False, line_terminator='\n')

        before = s['created_utc']
        submissions_lst.append(s)

        if scrapped == submission_to_crawl:
            break
        
    # Write submissions. Either create or append to csv
    df = json_normalize(submissions_lst)
    with open(join(output_dir, "submissions.csv"), 'a', encoding='utf-8') as f:
        df.to_csv(f, index = False, header=f.tell()==0, line_terminator='\n')
        
    scrapped += len(submissions_lst)

    if scrapped == submission_to_crawl:
        print('Finished!')
        exit()