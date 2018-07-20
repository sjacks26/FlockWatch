import datetime

FlockWatch_scheduling = {
    'repeat': True,
    'repeat_interval': datetime.time(hour=6,minute=0)
}

data_source = {
    "mongo": True,
    "csv": False,
    "mongo_details": {
        "AUTH": True,
        "username": "USERNAME",
        "password": "PASSWORD",
        #"database_name": "ConspiracyTheories_5b434d8a7315ac7dce057f13",
        "collection_name": "tweets",
        "field_name_for_date_restriction": "created_ts",
        "ignore_RTs": True
    },
    "csv_details": {
        "path": 'path/to/csv',
        "date_column_name": 'comment_created_at'
    },
    "collection_terms": None #['Trump', 'America', 'maga']              # If data come from mongo, FlockWatch uses "collection_name" to automatically find the collection terms
}

time_interval = [datetime.timedelta(hours=24)]
start_time = False # datetime.datetime(2017,11,19,23,59)
minimum_token_length = 3
trending = True
trending_threshold = 20                                             # This isn't actually a percent. The trending metric is (t2-t1)/((t2+t1)/2)*100. 66.67 = twice as many occurrences in t2 as t1. The max value is 200, which is always the value if a term appeared zero times in t1 and 1 time or more in t2.
co_occurrence = True
co_occurrence_threshold = 1/20                                      # This is a ratio representing how frequently words need to co-occur to be reported
repeat_reported_terms = {
    "repeat_limit": False,
    "limit": 4
}
report_context = True
context_examples = 5

stopword_file = False # '../stops.txt'
log_folder = './log/'

notification_email_recipients = ["user@email.com"]
account_to_send_emails = 'gmail.addy'                               # must be a gmail account. Don't include "@gmail.com"
password_to_send_emails = 'gmail.addy.password'
email_server = ("smtp.gmail.com", 587)