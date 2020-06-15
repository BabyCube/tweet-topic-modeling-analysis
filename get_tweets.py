import json
import twitter
import os
import argparse

"""
fetch tweets with keywords provided and output fected tweets as a JSON file

please alter the query string (line 40) to get the search result you want
"""

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--output_file_name', type=str,
                    help='Name for output file',
                    default='hash_tag_covid.json')
args = parser.parse_args()

# set the secret in your ENV
API_KEY = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
ACCESS_SECRET = os.environ.get("ACCESS_SECRET")

api = twitter.Api(consumer_key=API_KEY,
                  consumer_secret=API_SECRET,
                  access_token_key=ACCESS_TOKEN,
                  access_token_secret=ACCESS_SECRET)

# rate limit: must be less than 180 per 15 min window
# Search within hashtags of Twitter posts
# refer to https://developer.twitter.com/en/docs/tweets/rules-and-filtering/overview/standard-operators
# criteria: in English, full tweet, no retweets

# get tweets with hashtag COVID
query_str = "q=%23covid%20-filter%3Aretweets&src=typed_query&result_type=recent&lang=en&tweet_mode=extended&count=100"

results = api.GetSearch(raw_query=query_str)

tweets = []
min_tweetID = 999999999999999999999 # keep track of the min tweet ID, avoiding duplicate tweets from being fetched twice
max_tweetID = 0

for this_tweet_object in results:
    tweet_dict = {}
    tweet_dict["id"] = this_tweet_object.id
    tweet_dict["time"] = this_tweet_object.created_at
    tweet_dict["text"] = this_tweet_object.full_text
    tweet_dict["hashtags"] = []
    for hashtag in this_tweet_object.hashtags:
        tweet_dict["hashtags"].append(hashtag.text)
    if this_tweet_object.id < min_tweetID:
        min_tweetID = this_tweet_object.id
    if this_tweet_object.id > max_tweetID:
        max_tweetID = this_tweet_object.id
    tweets.append(tweet_dict)

# get more tweets
request_count = 0
while request_count < 165:
    print ("Current Request Count: ", request_count)
    min_tweetID = min_tweetID - 1 # decrement to avoid getting the same tweet
    results = api.GetSearch(raw_query=query_str + "&max_id=" + str(min_tweetID))
    request_count = request_count + 1
    for this_tweet_object in results:
        tweet_dict = {}
        tweet_dict["id"] = this_tweet_object.id
        tweet_dict["time"] = this_tweet_object.created_at
        tweet_dict["text"] = this_tweet_object.full_text
        tweet_dict["hashtags"] = []
        for hashtag in this_tweet_object.hashtags:
            tweet_dict["hashtags"].append(hashtag.text)
        if this_tweet_object.id < min_tweetID:
            min_tweetID = this_tweet_object.id
        tweets.append(tweet_dict)

with open(args.output_file_name, 'w') as wf:
    wf.write(
            '' +
            '\n'.join(json.dumps(i) for i in tweets) +
            '\n')

print (max_tweetID, min_tweetID)
