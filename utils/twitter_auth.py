import requests
from requests_oauthlib import OAuth1
import twitter
import json

with open('../../_auth/twitter.json') as f:
    auth_params = json.load(f)

def connect_old():
    """using OAuth1"""
    
    # Creating an OAuth Client connection
    auth = OAuth1 (
        auth_params['app_key'],
        auth_params['app_secret'],
        auth_params['oauth_token'],
        auth_params['oauth_token_secret']
    )

    # url according to twitter API
    url_rest = "https://api.twitter.com/1.1/search/tweets.json"

    return auth, url_rest

def connect():
    """Using python-twitter API"""
    api = twitter.Api(consumer_key=auth_params['app_key'],
        consumer_secret=auth_params['app_secret'],
        access_token_key=auth_params['oauth_token'],
        access_token_secret=auth_params['oauth_token_secret'],
        tweet_mode='extended')
    return api