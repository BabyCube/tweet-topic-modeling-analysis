import json
import re
import argparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer

"""
before running this script file, please make sure you have already generated the tweets JSON file using get_tweets

this script will create another JSON file and each tweet JSON object will now have a sentiment score attached
"""
parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--path', type=str,
                    help='File path to COVID-19 tweets JSON',
                    default='hash_tag_covid.json')
parser.add_argument('--output_file_name', type=str,
                    help='Name for output file',
                    default='hash_tag_covid_processed.json')
parser.add_argument('--keep_hashtag', type=bool,
                    help='Keep hashtags within tweets when performing analysis?',
                    default=True)
args = parser.parse_args()
vander = SentimentIntensityAnalyzer()

def nltk_sentiment_generator(tweet_content):
    sentiment_result = vander.polarity_scores(tweet_content)
    print (sentiment_result)
    return sentiment_result['compound'] # return only compound score


def simple_cleanup(tweet_content):
    # strip out links and unicode char (such as '\ud83c\udde9\ud83c\uddea') from tweet content
    tweet_content = re.sub(r'[^\x00-\x7F]+','', tweet_content)
    tweet_content = re.sub(r'http\S+', '', tweet_content)
    return tweet_content


def strip_hashtags(tweet_content, hashtag_list):
    for tag in hashtag_list:
        tweet_content = tweet_content.replace("#"+tag, '#')
    return tweet_content


def generate_output(object_dict_list, output_file):
    with open(output_file, 'w') as wf:
        wf.write(
            '' +
            '\n'.join(json.dumps(i) for i in object_dict_list) +
            '\n')
    return True

if __name__ == "__main__":
    processed_tweets = []
    progress_count = 0
    tweets_file = open(args.path, "r")

    if args.keep_hashtag:
        hashtag = True
    else:
        hashtag = False

    # process every tweet in original JSON by clean up text and generate sentiment
    for tweet_object in tweets_file:
        processed_tweet_content = {} # store everything processed here for this tweet
        tweet_object = json.loads(tweet_object.strip())
        tweet_content = tweet_object["text"]
        tweet_content = simple_cleanup(tweet_content)
        print(tweet_content)
        if not hashtag:
            tweet_content = strip_hashtags(tweet_content, tweet_object["hashtags"])
        sentiment = nltk_sentiment_generator(tweet_content)
        processed_tweet_content = tweet_object
        processed_tweet_content["SIA_sentiment"] = sentiment
        processed_tweet_content["text"] = tweet_content
        print(processed_tweet_content)
        progress_count = progress_count + 1
        print("___________")
        processed_tweets.append(processed_tweet_content)

    # generate a new JSON file with sentiment
    generate_output(processed_tweets, args.output_file_name)

    print ("PROCESS COMPLETED")