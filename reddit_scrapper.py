import requests
import time
import json
from datetime import datetime, timedelta
from pandas.io.json import json_normalize
import pandas
from os.path import join
from pathlib import Path

def get_comments(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/comment/search/", params=kwargs)
    data = r.json()['data']
    return data

def get_submissions(**kwargs):
    r = requests.get("https://api.pushshift.io/reddit/submission/search/", params=kwargs)
    #print(r)
    data = r.json()['data']
    return data

def get_comment_ids_of_submission(submission_id):
    r = requests.get("https://api.pushshift.io/reddit/submission/comment_ids/" + str(submission_id))
    comment_ids = r.json()['data']
    return comment_ids

# Scrapping parameters
submission_to_crawl = 10000
subreddit = 'NatureIsFuckingLit'
#days_to_crawl = 7
#after=(datetime.now() - timedelta(days=days_to_crawl)).timestamp()
before=int((datetime.now() - timedelta(days=1)).timestamp())
filter_submissions = ['id', 'title', 'author', 'score', 'created_utc', 'num_comments', 'is_video', 'url']
filter_comments = []

# Save directory
output_dir = 'database_{:d}'.format(before)
images_dir = join(output_dir, 'images')
comments_dir = join(output_dir, 'comments')
Path(images_dir).mkdir(parents=True, exist_ok=True)
Path(comments_dir).mkdir(parents=True, exist_ok=True)

# Save all scrapped submissions in this list
submissions_lst = []

while True:
    submissions = get_submissions(subreddit=subreddit, 
                                  before=before, 
                                  filter=filter_submissions, 
                                  num_comments=' >10',
                                  is_video='false',
                                  score='>50',
                                  size = 500
                                  )
    if not submissions: break

    for s in submissions:
        #print(s)
        #print(s)
        
        # Download image
        image_url = s['url']
        if '.png' in image_url:
            extension = '.png'
        elif '.jpg' in image_url or '.jpeg' in image_url:
            extension = '.jpeg'
        elif 'imgur' in image_url and not '.gifv' in image_url:
            image_url += '.jpeg'
            extension = '.jpeg'
        else:
            continue
        
        # TODO: make asynchronous
        r = requests.get(s['url'])
        if(r.status_code == 200):
            with open(join(images_dir, s['id'] + extension),"bx") as f:
                f.write(r.content)
        else:
            continue
        
        submissions_lst.append(s)
        
        # Crawl comments
        # TODO: Bundle more together
        #comment_ids = get_comment_ids_of_submission(s['id'])
        #print(s['id'])
        #print(comment_ids)
        #comments = get_comments(ids=comment_ids)
        #print(comments[0])
        
      
        #print(json.dumps(s, indent=4, sort_keys=True))
         # This will keep track of your position for the next call in the while loop
        before = s['created_utc']

    # sort_type = 'score'

    # This will then pass the ids collected from Pushshift and query Reddit's API for the most up to date information
    #comments = get_comments_from_reddit_api(comment_ids,author)
    #for comment in comments:
    #    comment = comment['data']
        # Do stuff with the comments (this will print out a JSON blob for each comment)
    #    comment_json = json.dumps(comment,ensure_ascii=True,sort_keys=True)
    #    print(comment_json)