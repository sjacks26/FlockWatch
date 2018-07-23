import nltk
import pandas as pd
import time
import datetime
from scipy import sparse, stats
import logging
import sys
import os
import json
import pymongo
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg                            # I might want to make the config file a CLI argument rather than hard coded in

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True, help="The name of the data collection being FlockWatched. If your data comes from STACK, this should be the STACK project name.")
args = vars(ap.parse_args())

collection_name = args['name']
log_file_name = './' + collection_name + '.log'

logging.basicConfig(filename=log_file_name,filemode='a+',level=logging.DEBUG, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")


def get_collection_terms():
    if cfg.data_source['csv']:
        collection_terms = cfg.data_source['collection_terms']
    elif cfg.data_source['mongo']:
        mongoClient = pymongo.MongoClient()
        if cfg.data_source["mongo_details"]["AUTH"]:
            mongoClient.admin.authenticate(cfg.data_source["mongo_details"]["username"], cfg.data_source["mongo_details"]['password'])
        mongo_dbs = mongoClient.database_names()
        collection_term_db = [d for d in mongo_dbs if (collection_name in d and 'Config' in d)][0]
        collection_term_db = mongoClient[collection_term_db]
        collection_term_col = collection_term_db[collection_term_db.collection_names()[0]]
        collectors = collection_term_col.find({"network": "twitter"})
        collection_terms = []
        for collector in collectors:
            terms_list = collector["terms_list"]
            for term in terms_list:
                collection_terms.append(term["term"])
    collection_terms = [c.lower() for c in collection_terms]
    return collection_terms

collection_terms = get_collection_terms()


def build_stopwords():
    stops = list(nltk.corpus.stopwords.words('english'))
    if cfg.stopword_file:
        with open(cfg.stopword_file, 'r') as f:
            additional_stops = f.read().splitlines()
        stops.extend(additional_stops)
    stops = [w.lower() for w in stops]
    return stops

stops = build_stopwords()
stops_w_collection_terms = stops.copy()
stops_w_collection_terms.extend(collection_terms)


def find_text(interval):
    if cfg.start_time:
        start_time = cfg.start_time
    elif not cfg.start_time:
        start_time = datetime.datetime.now().utcnow()
    logging.debug('Start time: {}'.format(str(start_time)))
    logging.debug('Interval: {}'.format(str(interval)))
    early_text_start = start_time - (2 * interval)
    text_bridge_time = start_time - interval
    if cfg.data_source['mongo']:
        logging.info("Getting text from Mongo.")
        mongoClient = pymongo.MongoClient()
        if cfg.data_source['mongo_details']['AUTH']:
            mongoClient.admin.authenticate(cfg.data_source['mongo_details']["username"], cfg.data_source['mongo_details']['password'])
        mongo_dbs = mongoClient.database_names()
        data_db = [d for d in mongo_dbs if (collection_name in d and '_' in d)][0]
        data_db = mongoClient[data_db]
        data_col = data_db[cfg.data_source['mongo_details']["collection_name"]]
        if cfg.data_source['mongo_details']['ignore_RTs']:
            logging.info("Ignoring retweets.")
            early_text = list(data_col.find({"retweeted_status": {"$exists": False}, cfg.data_source['mongo_details']['field_name_for_date_restriction']: {"$gte": early_text_start, "$lt": text_bridge_time}}, projection={"_id": 0, "stack_vars.full_tweet.full_text": 1}))
            late_text = list(data_col.find({"retweeted_status": {"$exists": False}, cfg.data_source['mongo_details']['field_name_for_date_restriction']: {"$gte": text_bridge_time, "$lt": start_time}}, projection={"_id": 0, "stack_vars.full_tweet.full_text": 1}))
        elif not cfg.data_source['mongo_details']['ignore_RTs']:
            early_text = list(data_col.find({cfg.data_source['mongo_details']['field_name_for_date_restriction']: {"$gte": early_text_start, "$lt": text_bridge_time}}, projection={"_id":0, "stack_vars.full_tweet.full_text":1}))
            late_text = list(data_col.find({cfg.data_source['mongo_details']['field_name_for_date_restriction']: {"$gte": text_bridge_time, "$lt": start_time}}, projection={"_id": 0, "stack_vars.full_tweet.full_text": 1}))
        early_text = pd.DataFrame([t["stack_vars"]["full_tweet"]["full_text"] for t in early_text], columns=["text"])
        late_text = pd.DataFrame([t["stack_vars"]["full_tweet"]["full_text"] for t in late_text], columns=["text"])
    elif cfg.data_source['csv']:
        logging.info("Getting text from CSV.")
        text = pd.read_csv(cfg.data_source['csv_details']['path'], keep_default_na=False, parse_dates=[cfg.data_source['csv_details']['comment_created_at']])
        early_text = text[(pd.to_datetime(text['comment_created_at']) >= early_text_start) & (pd.to_datetime(text['comment_created_at']) < text_bridge_time)]
        late_text = text[(pd.to_datetime(text['comment_created_at']) >= text_bridge_time) & (pd.to_datetime(text['comment_created_at']) < start_time)]
    if len(early_text) == 0 or len(late_text) == 0:
        logging.warning("Based on your desired interval, one of the sets of messages does not contain any messages. Try a different interval or wait for more data.")
        sys.exit()
    logging.info("Found two datasets. The first (from {0} to {1}) contains {2} messages. The second (from {3} to {4}) contains {5} messages.".format(early_text_start, text_bridge_time, len(early_text), text_bridge_time, start_time, len(late_text)))
    return early_text, late_text


def build_word_frequency(text1, text2):
    """
    Once we have two sets of text data, the first step is to get word frequencies for each set.
    This function returns a dataframe with word counts for all words that appear in both sets of text
    """
    tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
    tokens1 = [tknzr.tokenize(text) for text in text1.iloc[:,0]]
    tokens2 = [tknzr.tokenize(text) for text in text2.iloc[:,0]]
    tokens1 = [w.strip('#@') for t in tokens1 for w in t if not w.strip('#@') in stops_w_collection_terms]
    tokens2 = [w.strip('#@') for t in tokens2 for w in t if not w.strip('#@') in stops_w_collection_terms]
    tokens1 = [t for t in tokens1 if t.isalnum()]
    tokens2 = [t for t in tokens2 if t.isalnum()]
    tokens1 = [t for t in tokens1 if len(t) >= cfg.minimum_token_length]
    tokens2 = [t for t in tokens2 if len(t) >= cfg.minimum_token_length]
    logging.info("Found {0} tokens after filtering in early text. Found {1} tokens after filtering in late text.".format(len(tokens1), len(tokens2)))
    counts1 = nltk.FreqDist(tokens1)
    counts2 = nltk.FreqDist(tokens2)
    counts_df_columns = ['word', 'count1', 'count2']
    counts_list = []
    for c in counts2:
        if c in counts1 and counts2[c] > (len(text2)/100):
            counts_list.append([c, counts1[c], counts2[c]])
        elif c not in counts1 and counts2[c] > (len(text2)/100):                    # The elif lets us get terms that didn't appear at all in the first set of text
            counts_list.append([c, 0, counts2[c]])                          # We give the count of a term that doesn't appear in the early text a sub-one number so we can still get a rate of change. The value needs to be tuned to figure out how many times a term should appear in the second text if it never appeared in the first text in order to appear in the trending terms list
    counts_df = pd.DataFrame(counts_list, columns = counts_df_columns)
    counts_df.sort_values(by=['count2','count1'], ascending=False, inplace=True)
    logging.info("Got counts for {} unique tokens".format(counts_df.shape[0]))
    return counts_df


def find_trending_context(text2, trending_term, number_examples=cfg.context_examples):
    text = text2['text'].str.lower()
    messages_with_trending_term = text[text.str.contains(trending_term)]
    if messages_with_trending_term.shape[0] > number_examples:
        context = list(messages_with_trending_term.sample(number_examples))
    elif messages_with_trending_term.shape[0] <= number_examples:
        context = list(messages_with_trending_term)
    return context


def find_trending_words(text1, text2):
    trending_df = build_word_frequency(text1, text2)
    trending_df['rate_of_change'] = (trending_df['count2'] - trending_df['count1'])/((trending_df['count2'] + trending_df['count1']) / 2) * 100
    trending_df = trending_df[trending_df['rate_of_change'] > cfg.trending_threshold]
    trending_df.sort_values(by=['rate_of_change', 'count2'], ascending=False, inplace=True)
    trending_df.reset_index(drop=True, inplace=True)
    logging.info("Found {0} trending terms.".format(trending_df.shape[0]))
    context = {}
    if cfg.report_context:
        for word in trending_df['word']:
            context[word] = find_trending_context(text2, word)
    elif not cfg.report_context:
        context = None
    return trending_df, context


def build_cooccurrence_matrix(text):
    """
    I'm not sure whether I want to use text separated by time. Might just use all text together or just the most recent text.
    Code based on a sample by Carl McCaffrey of UCD
    """
    text = text['text'].str.replace('@','').str.replace('#','').str.lower()
    text.reset_index(inplace=True, drop=True)
    tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
    tokens = [tknzr.tokenize(t) for t in text]
    clean_texts = [w for t in tokens for w in t if (not w in stops_w_collection_terms and w.isalnum() and len(w) >= cfg.minimum_token_length)]
    clean_tokens = list(set(clean_texts))
    logging.info("Found {0} tokens that might occur with collection terms.".format(len(clean_tokens)))
    # one_cooccurrence_value = 1/len(tokens)
    '''
    one_cooccurrence_value assumes that the cooccurrence rate we care about is relative to the whole body of messages, not to the number of times a particular term appears. I don't think that's right.
    The next block changes this. Instead, the value per cooccurrence between a token and a collection term is 1 divided by the number of documents that contain that collection term.
    '''
    cooccurrence_values = {}
    for c in collection_terms:
        count = text.str.contains(c).sum()
        if count > (.001 * len(text)) and 1/count < (cfg.co_occurrence_threshold/10):
            cooccurrence_values[c] = 1/count
            logging.info("Value for one co-occurrence with {0} is {1}.".format(c, cooccurrence_values[c]))
        elif count == 0:
            logging.info("No messages with {0} found. Not looking for co-occurrences with this term.".format(c))
        elif count > 0 and 1/count >= (cfg.co_occurrence_threshold/10):
            logging.info("Too few messages with {0} found. Not looking for co-occurrences with this term.".format(c))
    good_collects = list(cooccurrence_values.keys())
    good_collect_ngrams = [f for f in good_collects if ' ' in f]
    co_matrix = sparse.dok_matrix((len(good_collects), len(clean_tokens)))
    clean_messages = [[t for t in w if (not t in stops and t.isalnum() and len(t) >= cfg.minimum_token_length)] for w in tokens]
    message_time = []
    for i in range(0, len(clean_messages)):
        start_time = time.time()
        clean_message = clean_messages[i]
        present_collection_terms = list(set(good_collects).intersection(clean_message))
        present_collect_ngrams = [f for f in good_collect_ngrams if f in text[i]]
        present_collection_terms.extend(present_collect_ngrams)
        collection_term_indexes = {}
        message = list(set(clean_message))
        for c in present_collection_terms:
            c_term_index = good_collects.index(c)
            collection_term_indexes[c] = c_term_index
            if c in present_collect_ngrams:
                c_tokens = c.split(' ')
                for t in c_tokens:
                    if t in message:
                        message.remove(t)
        for token in message:
            if token in clean_tokens:
                token_index = clean_tokens.index(token)
                for c in collection_term_indexes:
                    co_matrix[collection_term_indexes[c], token_index] += cooccurrence_values[c]
        stop_time = time.time()
        duration = stop_time - start_time
        message_time.append(duration)
    co_matrix_df = pd.DataFrame(co_matrix.todense())
    co_matrix_df.columns = clean_tokens
    new_index = dict(enumerate(good_collects))
    co_matrix_df = co_matrix_df.rename(index=new_index)
    co_matrix_df = co_matrix_df.transpose()

    describe_time = stats.describe(message_time)
    logging.debug("Mean time to calculate co-occurrence rates per message: {}".format(describe_time.mean))
    time_plot = plt.figure()
    plt.hist(message_time, log=True)
    fig_name = 'time_plot'
    plt.savefig(fig_name)
    plt.close('all')
    message_time_without_outliers = [f for f in message_time if f < (10 * describe_time.mean)]
    zoomed_time_plot = plt.figure()
    plt.hist(message_time_without_outliers, log=True)
    fig_name = 'zommed_time_plot'
    plt.savefig(fig_name)
    plt.close('all')
    return co_matrix_df


def find_co_occurrence_context(text, co_occurrence_pair, number_examples=cfg.context_examples):
    text = text['text'].str.lower()
    messages_with_co_occurence = text[text.str.contains(co_occurrence_pair[0]) & text.str.contains(co_occurrence_pair[1])]
    if messages_with_co_occurence.shape[0] > number_examples:
        context = list(messages_with_co_occurence.sample(number_examples))
    elif messages_with_co_occurence.shape[0] <= number_examples:
        context = list(messages_with_co_occurence)
    return context


def get_co_occurrence_pairs(text):
    """
    This function builds a list of co-occurring words to report. It only reports pairs of words if they co-occur above the threshold set in co_occurrence_threshold
    Code based on a sample by Carl McCaffrey of UCD
    """
    co_matrix_df = build_cooccurrence_matrix(text)
    pair_list = []
    matrix = sparse.coo_matrix(co_matrix_df.values.T)
    for i,j,rate in zip(matrix.row, matrix.col, matrix.data):
        if rate > cfg.co_occurrence_threshold:
            pair_list.append([co_matrix_df.columns[i], co_matrix_df.index[j], rate])
    if len(pair_list) > 0:
        pairs = pd.DataFrame(pair_list)
        pairs.columns = ['collection_term', 'co-occurring_term', 'rate']
        pairs.sort_values(by=['rate'], ascending=False, inplace=True)
        pairs.reset_index(drop=True, inplace=True)
        logging.info("Found {0} tokens that co-occur with collection terms at a rate of at least {1}.".format(pairs.shape[0], cfg.co_occurrence_threshold))
        context = {}
        if cfg.report_context:
            for pair in pair_list:
                pair = pair[:2]
                context[' '.join(pair)] = find_co_occurrence_context(text, pair)
        elif not cfg.report_context:
            context = None
    elif len(pair_list) == 0:
        pairs = pd.DataFrame(columns=['collection_term', 'co-occurring_term', 'rate'])
        logging.info("Found zero tokens that co-occur with collection terms at a rate of at least {0}.".format(cfg.co_occurrence_threshold))
        context = None
    return pairs, context


def limit_repeat_reports(text):
    if cfg.repeat_reported_terms["repeat_limit"]:
        columns_to_check = ['word', 'co-occurring_term']
        terms_to_ignore = []
        log_file = os.path.join(cfg.log_folder, collection_name, 'term_history.json')
        if os.path.isfile(log_file):
            with open(log_file, 'r') as f:
                term_history = json.load(f)
            for term in term_history:
                if term_history[term] > cfg.repeat_reported_terms['limit']:
                    terms_to_ignore.append(term)
            for c in columns_to_check:
                if c in text.columns:
                    text = text[~trending_df[c].isin(terms_to_ignore)]
        logging.info("Removed recommended terms that have been recommended more than {0} times in the past.".format(repeat_reported_terms['limit']))
    else:
        pass
    return text


def write_trending_report(trending_df, log_dir):
    trending_log = os.path.join(log_dir, 'trending_terms.csv')
    trending_df.to_csv(trending_log, index=False)
    logging.info("Wrote trending terms report to {0}.".format(trending_log))
    return trending_log


def write_cooccurrence_report(cooccurrence_df, log_dir):
    cooccurrence_log = os.path.join(log_dir, 'co-occurring_terms.csv')
    cooccurrence_df.to_csv(cooccurrence_log, index=False)
    logging.info("Wrote co-occurring terms report to {0}.".format(cooccurrence_log))
    return cooccurrence_log


def log_term_recommendations(text):
    term_history = {}
    columns = text.columns
    if 'word' in columns:
        terms = list(set(list(text['word'])))
    elif 'co-occurring_term' in columns:
        terms = list(set(list(text['co-occurring_term'])))
    log_file = os.path.join(cfg.log_folder, collection_name, 'term_history.json')
    if os.path.isfile(log_file):
        with open(log_file, 'r') as f:
            term_history = json.load(f)
            os.remove(log_file)
    for term in terms:
        if term in term_history:
            term_history[term] += 1
        elif not term in term_history:
            term_history[term] = 1
    with open(log_file, 'w') as f:
        json.dump(term_history, f, indent="\t", sort_keys=True)
    logging.info("Updated recommended term history log at {}.".format(log_file))


def email_notifications(cooccurrence_log, trending_log, trending_df, cooccurrence_df):
    if not type(cfg.notification_email_recipients) is list:
        raise Exception("Email recipients must be in a list")
    email = EmailMessage()
    cooccurrence_pairs = cooccurrence_df.shape[0]
    trending_terms = trending_df.shape[0]
    sample_trending = list(trending_df['word'].head())
    email_text = "FlockWatch found {0} trending terms and {1} words that co-occur with collection terms.\nSee the trending report ({2}) and the co-occurrence report ({3}) for full details.\n\n".format(trending_terms, cooccurrence_pairs, trending_log, cooccurrence_log)
    email_text += "Here are a few of the most trending terms:\n\t{0}".format('\n\t'.join(sample_trending))
    email.set_content(email_text)
    email['Subject'] = "FlockWatch report for {0}".format(collection_name)
    email['From'] = cfg.account_to_send_emails + '@gmail.com'
    email['To'] = ", ".join(cfg.notification_email_recipients)
    server = smtplib.SMTP(cfg.email_server[0], cfg.email_server[1])
    server.starttls()
    server.login(cfg.account_to_send_emails, cfg.password_to_send_emails)
    server.sendmail(email['From'], cfg.notification_email_recipients, email.as_string())
    server.quit()


def main():
    duration = 0
    logging.info("Searching for new terms to be used alongside the existing collection criteria ({0}).".format(collection_terms))
    log_folder = os.path.join(cfg.log_folder, collection_name, str(datetime.date.today()), str(datetime.datetime.now().time().replace(microsecond=0)))
    os.makedirs(log_folder, exist_ok=True)
    for interval in cfg.time_interval:
        logging.info("Running based on time interval of {0}.".format(str(interval)))
        text1, text2 = find_text(interval)
        if cfg.trending:
            trending_start = time.process_time()
            trending_df, trending_context = find_trending_words(text1, text2)
            '''
            I haven't figured out what to do with trending_context yet. No idea how to report this.
            '''
            trending_stop = time.process_time()
            trending_time = trending_stop - trending_start
            logging.debug("Took {} seconds to calculate trending terms".format(trending_time))
            trending_df = limit_repeat_reports(trending_df)
            log_term_recommendations(trending_df)
            if trending_df.shape[0] > 0:
                trending_log = write_trending_report(trending_df, log_folder)
            elif trending_df.shape[0] == 0:
                trending_log = "No trending log"
        elif not cfg.trending:
            logging.info("Not searching for trending terms.")
            trending_log = "Not searching for trending terms"
            trending_df = pd.DataFrame(columns=['word'])
            trending_time = 0
        duration += trending_time
        if cfg.co_occurrence:
            report_pairs_start = time.process_time()
            cooccurrence_df, cooccurrence_context = get_co_occurrence_pairs(text1.append(text2))
            report_pairs_stop = time.process_time()
            report_pairs_time = report_pairs_stop - report_pairs_start
            logging.debug("Took {} seconds to find co-occurrence pairs".format(report_pairs_time))
            cooccurrence_df = limit_repeat_reports(cooccurrence_df)
            log_term_recommendations(cooccurrence_df)
            if cooccurrence_df.shape[0] > 0:
                cooccurrence_log = write_cooccurrence_report(cooccurrence_df, log_folder)
            elif cooccurrence_df.shape[0] == 0:
                cooccurrence_log = "No co-occurrence log"
        elif not cfg.co_occurrence:
            logging.info("Not searching for co-occurring terms.")
            cooccurrence_log = "Not searching for co-occurring terms."
            cooccurrence_df = pd.DataFrame(columns=['word'])
            report_pairs_time = 0
        duration += report_pairs_time
        email_notifications(cooccurrence_log, trending_log, trending_df, cooccurrence_df)
    return duration

running = True
while running:
    duration = main()
    if not cfg.FlockWatch_scheduling['repeat']:
        running = False
    elif cfg.FlockWatch_scheduling['repeat']:
        sleep_time = (cfg.FlockWatch_scheduling['repeat_interval'].hour * 60 * 60) + (cfg.FlockWatch_scheduling['repeat_interval'].minute * 60) - duration
        if sleep_time < 0:
            logging.warning("FlockWatch takes too long to complete with your parameters for it to run as frequently as you want. FlockWatch will run again as soon as it can.\n")
        elif sleep_time > 0:
            logging.info("FlockWatch is complete. Sleeping for {} seconds, then running again.\n".format(sleep_time))
            time.sleep(sleep_time)

"""
Next steps:
- Add functionality to run this on multiple independent collections on the same server and still be able to distinguish which FlockReport output belongs to which collection.
- Add option to report context for trending words and co-occurrences
    - Context generated.
    - Need to figure out how to report context.
"""

logging.info("Process finished\n")