import requests
import time
import json
from datetime import datetime, timedelta
from pandas import json_normalize
import pandas
from os.path import join
from pathlib import Path

urls = {
        'comments' : 'https://api.pushshift.io/reddit/comment/search/',
        'submissions' : 'https://api.pushshift.io/reddit/submission/search/',
        'comment_ids' : 'https://api.pushshift.io/reddit/submission/comment_ids/'
        }

def get(what, **kwargs):
    r = requests.get(urls[what], params=kwargs)
    try:
        data = r.json()['data']
    except:
        return None
    return data

def get_comment_ids_of_submission(submission_id):
    r = requests.get(urls['comment_ids'] + str(submission_id))
    try:
        comment_ids = r.json()['data']
    except:
        return None
    return comment_ids

# Program parameters
submission_to_crawl = 1000000
database = 'database_1596481650' # Set to None to start from today, else continues on specified database
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

if database is None:
    # Save directory
    #output_dir = 'database_{:d}'.format(before)
    before=int((datetime.now() - timedelta(days=1)).timestamp())
    output_dir = 'H:\\reddit\\' + 'database_{:d}'.format(before)
else:
    output_dir = 'H:\\reddit\\' + database
    df = pandas.read_csv(join(output_dir, "submissions.csv"))
    before=int(df['created_utc'].min())
    
comments_dir = join(output_dir, 'comments')
images_dir = join(output_dir, 'images')
comments_dir = join(output_dir, 'comments')
Path(images_dir).mkdir(parents=True, exist_ok=True)
Path(comments_dir).mkdir(parents=True, exist_ok=True)

scrapped = 0

while True:
    print('Scrapped {:d}/{:d} submissions...'.format(scrapped, submission_to_crawl))
    # Save all scrapped submissions in this list
    submissions_lst = []
    submissions = get('submissions', before=before, **submissions_params)
    if not submissions: 
        print('No submissions left!')     
        exit()

    for s in submissions:
        # Get image url
        image_url = s['url']
        if '.png' in image_url:
            extension = '.png'
        elif '.jpg' in image_url or '.jpeg' in image_url:
            extension = '.jpeg'
        elif 'imgur' in image_url and not '.gifv' in image_url:
            if not '.jpg' in image_url:
                image_url = image_url.replace('imgur', 'i.imgur')
                image_url += '.jpg'
            if 'm.i.imgur' in image_url:
                image_url = image_url.replace('m.i.imgur', 'i.imgur')
            extension = '.jpeg'
            #print(image_url)
        else:
            continue

        # Crawl comments
        # TODO: Async
        comment_ids = get_comment_ids_of_submission(s['id'])
        if comment_ids is None: continue
        comments = get('comments', ids=comment_ids, **comments_params)
        if comments is None: continue
        
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
            
        # Save image
        # TODO: Async
        r = requests.get(image_url)
        if(r.status_code == 200):
            with open(join(images_dir, s['id'] + extension),'wb') as f:
                f.write(r.content)
        else: continue

        before = s['created_utc']
        submissions_lst.append(s)

        scrapped += 1
        if scrapped == submission_to_crawl:
            break
        
    # Write submissions. Either create or append to csv
    df = json_normalize(submissions_lst)
    with open(join(output_dir, 'submissions.csv'), 'a', encoding='utf-8') as f:
        df.to_csv(f, index = False, header=f.tell()==0, line_terminator='\n')

    if scrapped == submission_to_crawl:
        print('Finished!')
        exit()