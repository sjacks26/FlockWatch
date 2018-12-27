try:
    import nltk
    import pandas as pd
    from pandas.io.json import json_normalize
    import time
    import datetime
    from scipy import sparse, stats
    import logging
    import sys
    import os
    import re
    import json
    import pymongo
    import pathlib
    from email.message import EmailMessage
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    import smtplib
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    logging.critical("Couldn't import a required module. Did you install using requirements.txt?")
    sys.exit()

import config as cfg                            # I might want to make the config file a CLI argument rather than hard coded in

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True, help="The name of the data collection being FlockWatched. If your data comes from STACK, this should be the STACK project name.")
args = vars(ap.parse_args())

collection_name = args['name']
log_file_name = str(pathlib.PurePath('.', collection_name + '.log'))

logging.basicConfig(filename=log_file_name,filemode='a+',level=logging.DEBUG, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")


def custom_alpha_filter(word):
    custom_filter = '[^A-Za-z0-9_\-]'
    filter_match = re.findall(custom_filter, word)
    if len(filter_match) == 0:
        return True
    else:
        return False


def get_collection_terms():
    if cfg.data_source['csv'] or cfg.data_source['json']:
        collection_terms = cfg.data_source['collection_terms']
        if not collection_terms:
            logging.critical("No collection terms were provided in the config file.")
            sys.exit()
    elif cfg.data_source['mongo']:
        try:
            mongoClient = pymongo.MongoClient()
        except:
            logging.critical("Mongo isn't running! Start Mongo or use CSV to give FlockWatch data.")
            sys.exit()
        if cfg.data_source["mongo_details"]["AUTH"]:
            try:
                mongoClient.admin.authenticate(cfg.data_source["mongo_details"]["username"], cfg.data_source["mongo_details"]['password'])
            except:
                logging.critical("Couldn't authenticate Mongo. Check the credentials and try again.")
                sys.exit()
        try:
            mongo_dbs = mongoClient.database_names()
            collection_term_db = [d for d in mongo_dbs if (collection_name in d and 'Config' in d)][0]
        except:
            logging.critical("Couldn't find a config db for the STACK project you provided.")
            sys.exit()
        collection_term_db = mongoClient[collection_term_db]
        collection_term_col = collection_term_db[collection_term_db.collection_names()[0]]
        collectors = collection_term_col.find({"network": "twitter"})
        collection_terms = []
        for collector in collectors:
            terms_list = collector["terms_list"]
            for term in terms_list:
                collection_terms.append(term["term"])
    collection_terms = [c.lower() for c in collection_terms]
    if len(collection_terms) == 0:
        if cfg.data_source['mongo']:
            logging.critical("No collection terms found from Mongo! Check STACK config db contents.")
            sys.exit()
        elif cfg.data_source['csv']:
            logging.critical("No collection terms found! Did you remember to provide them in the config file?")
            sys.exit()
        elif cfg.data_source['json']:
            logging.critical("No collection terms found! Did you remember to provide them in the config file?")
            sys.exit()
    return collection_terms


def build_stopwords():
    try:
        stops = list(nltk.corpus.stopwords.words('english'))
    except:
        logging.critical("NLTK stopwords aren't installed. Check README for instructions on how to install NLTK stopwords: https://github.com/sjacks26/FlockWatch#installation-and-setup")
        sys.exit()
    if cfg.stopword_file:
        try:
            with open(cfg.stopword_file, 'r') as f:
                additional_stops = f.read().splitlines()
            stops.extend(additional_stops)
        except:
            logging.warning("Couldn't read additional stopwords from file. Continuing with default stopwords.")
    stops = [w.lower() for w in stops]
    return stops


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
        try:
            data_db = [d for d in mongo_dbs if (collection_name in d and '_' in d)][0]
        except:
            logging.critical("Couldn't find a data db for the STACK project you provided.")
            sys.exit()
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
        try:
            text = pd.read_csv(cfg.data_source['csv_details']['path'], keep_default_na=False, parse_dates=[cfg.data_source['csv_details']['date_column_name']])
        except:
            logging.critical("Couldn't read the CSV file you specified. Did you provide the correct path?")
            sys.exit()
        try:
            text = text[[cfg.data_source['csv_details']['text_column_name'], cfg.data_source['csv_details']['date_column_name']]]
            text.columns = ['text', 'text_date']
        except:
            logging.critical("Couldn't find the columns named in the config file.")
            sys.exit()
        try:
            early_text = text[(pd.to_datetime(text['text_date']) >= early_text_start) & (pd.to_datetime(text['text_date']) < text_bridge_time)]
            late_text = text[(pd.to_datetime(text['text_date']) >= text_bridge_time) & (pd.to_datetime(text['text_date']) < start_time)]
        except:
            logging.warning("Couldn't parse the date column in the CSV file using pandas's built-in to_datetime function. Is it a valid timestamp? You might need to reformat that column to something that pandas can understand.")
            sys.exit()
    elif cfg.data_source['json']:
        logging.info("Getting text from JSON.")
        text = pd.DataFrame()
        try:
            with open(cfg.data_source['json_details']['path'], 'r') as f:
                text_data = f.readlines()
        except:
            logging.critical("Couldn't read the JSON file you specified. Did you provide the correct path?")
            sys.exit()
        try:
            for t in text_data:
                text_line = json.loads(t)
                text_line = json_normalize(text_line)
                text = text.append(text_line)
            text = text[[cfg.data_source['json_details']['text_key_name'], cfg.data_source['json_details']['date_key_name']]]
            text.columns = ['text', 'text_date']
        except:
            logging.critical("Couldn't find the keys named in the config file.")
            sys.exit()
        try:
            early_text = text[(pd.to_datetime(text['text_date']) >= early_text_start) & (pd.to_datetime(text['text_date']) < text_bridge_time)]
            late_text = text[(pd.to_datetime(text['text_date']) >= text_bridge_time) & (pd.to_datetime(text['text_date']) < start_time)]
        except:
            logging.warning("Couldn't parse the date value in the JSON file using panda's built-in to_datetime function. Is it a valid timestamp? You might need to reformat it to something that pandas can understand.")
            sys.exit()
    if len(early_text) == 0 or len(late_text) == 0:
        logging.warning("Based on your desired interval, one of the sets of messages does not contain any messages. Try a different interval or wait for more data.")
        sys.exit()
    logging.info("Found two datasets. The first (from {0} to {1}) contains {2} messages. The second (from {3} to {4}) contains {5} messages.".format(early_text_start, text_bridge_time, len(early_text), text_bridge_time, start_time, len(late_text)))
    return early_text, late_text


def build_word_frequency(text1, text2, stops_w_collection_terms):
    """
    Once we have two sets of text data, the first step is to get word frequencies for each set.
    This function returns a dataframe with word counts for all words that appear in both sets of text
    """
    tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
    tokens1 = [tknzr.tokenize(text) for text in text1['text']]
    tokens2 = [tknzr.tokenize(text) for text in text2['text']]
    if cfg.ignore_handles:
        tokens1 = [w.strip('#') for t in tokens1 for w in t if (not '@' in w and not w.strip('#') in stops_w_collection_terms)]
        tokens2 = [w.strip('#') for t in tokens2 for w in t if (not '@' in w and not w.strip('#') in stops_w_collection_terms)]
    elif not cfg.ignore_handles:
        tokens1 = [w.strip('#@') for t in tokens1 for w in t if not w.strip('#@') in stops_w_collection_terms]
        tokens2 = [w.strip('#@') for t in tokens2 for w in t if not w.strip('#@') in stops_w_collection_terms]
    tokens1 = [t for t in tokens1 if custom_alpha_filter(t)()]
    tokens2 = [t for t in tokens2 if custom_alpha_filter(t)()]
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
            counts_list.append([c, 0, counts2[c]])
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
    if len(context) == 0:
        context = None
    return context


def find_trending_words(text1, text2, stops_w_collection_terms):
    trending_df = build_word_frequency(text1, text2, stops_w_collection_terms)
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


def build_bigram_frequency(text1, text2):
    tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
    tokens1 = [tknzr.tokenize(text) for text in text1['text']]
    tokens2 = [tknzr.tokenize(text) for text in text2['text']]
    if cfg.ignore_handles:
        tokens1 = [w.strip('#') for t in tokens1 for w in t]
        tokens2 = [w.strip('#') for t in tokens2 for w in t]
    elif not cfg.ignore_handles:
        tokens1 = [w.strip('#@') for t in tokens1 for w in t]
        tokens2 = [w.strip('#@') for t in tokens2 for w in t]
    bigrams1  = list(nltk.bigrams(tokens1))
    bigrams2 = list(nltk.bigrams(tokens2))
    bigrams1 = [t for t in bigrams1 if not (t[0] in stops_w_collection_terms or t[1] in stops_w_collection_terms) and (custom_alpha_filter(t[0]) and custom_alpha_filter(t[1])) and ' '.join(t) not in stops_w_collection_terms and '@' not in ' '.join(t)]
    bigrams2 = [t for t in bigrams2 if not (t[0] in stops_w_collection_terms or t[1] in stops_w_collection_terms) and (custom_alpha_filter(t[0]) and custom_alpha_filter(t[1])) and ' '.join(t) not in stops_w_collection_terms and '@' not in ' '.join(t)]
    logging.info("Found {0} bigrams after filtering in early text. Found {1} bigrams after filtering in late text.".format(len(bigrams1), len(bigrams2)))
    counts1 = nltk.FreqDist(bigrams1)
    counts2 = nltk.FreqDist(bigrams2)
    counts_df_columns = ['bigram', 'count1', 'count2']
    counts_list = []
    for c in counts2:
        if c in counts1 and counts2[c] > (len(text2) / 100):
            counts_list.append([' '.join(c), counts1[c], counts2[c]])
        elif c not in counts1 and counts2[c] > (len(text2) / 100):  # The elif lets us get terms that didn't appear at all in the first set of text
            counts_list.append([' '.join(c), 0, counts2[c]])
    counts_df = pd.DataFrame(counts_list, columns=counts_df_columns)
    counts_df.sort_values(by=['count2', 'count1'], ascending=False, inplace=True)
    logging.info("Got counts for {} unique bigrams".format(counts_df.shape[0]))
    return counts_df


def find_trending_bigrams(text1, text2):
    trending_df = build_bigram_frequency(text1, text2)
    trending_df['rate_of_change'] = (trending_df['count2'] - trending_df['count1'])/((trending_df['count2'] + trending_df['count1']) / 2) * 100
    trending_df = trending_df[trending_df['rate_of_change'] > cfg.trending_threshold]
    trending_df.sort_values(by=['rate_of_change', 'count2'], ascending=False, inplace=True)
    trending_df.reset_index(drop=True, inplace=True)
    logging.info("Found {0} trending bigrams.".format(trending_df.shape[0]))
    context = {}
    if cfg.report_context:
        for bigram in trending_df['bigram']:
            context[bigram] = find_trending_bigram_context(text2, bigram)
    elif not cfg.report_context:
        context = None
    return trending_df, context


def find_trending_bigram_context(text2, trending_bigram, number_examples=cfg.context_examples):
    '''
    Need to build this function
    '''
    return


def build_cooccurrence_matrix(text):
    """
    Code based on a sample by Carl McCaffrey of UCD
    """
    if cfg.ignore_handles:
        text = text['text'].str.replace('#', '').str.lower()
    elif not cfg.ignore_handles:
        text = text['text'].str.replace('@','').str.replace('#','').str.lower()
    text.reset_index(inplace=True, drop=True)
    tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
    tokens = [tknzr.tokenize(t) for t in text]
    clean_texts = [w for t in tokens for w in t if (not w in stops_w_collection_terms and custom_alpha_filter(w) and len(w) >= cfg.minimum_token_length)]
    clean_tokens = list(set(clean_texts))
    logging.info("Found {0} tokens that might occur with collection terms.".format(len(clean_tokens)))
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
    logging.info("Searching for tokens that co-occur with {0} collection terms.".format(len(good_collects)))
    good_collect_ngrams = [f for f in good_collects if (' ' in f or '-' in f)]
    co_matrix = sparse.dok_matrix((len(good_collects), len(clean_tokens)))
    clean_messages = [[t for t in w if (not t in stops and custom_alpha_filter(t)() and len(t) >= cfg.minimum_token_length)] for w in tokens]
    message_time = []
    if len(good_collects) > 0:
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
                    c_tokens = c.replace('-', ' ').split(' ')
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
        fig_name = 'zoomed_time_plot'
        plt.savefig(fig_name)
        plt.close('all')
    elif len(good_collects) == 0:
        co_matrix_df = pd.DataFrame(co_matrix.todense())
        co_matrix_df.columns = clean_tokens
        co_matrix_df = co_matrix_df.transpose()
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
        """
        How do I add bigrams to this?
        """
        terms_to_ignore = []
        log_file = pathlib.PurePath(cfg.log_folder, collection_name, 'term_history.json')
        if os.path.isfile(log_file):
            with open(log_file, 'r') as f:
                term_history = json.load(f)
            for term in term_history:
                if term_history[term] > cfg.repeat_reported_terms['limit']:
                    terms_to_ignore.append(term)
            for c in columns_to_check:
                if c in text.columns:
                    text = text[~text[c].isin(terms_to_ignore)]
        logging.info("Removed recommended terms that have been recommended more than {0} times in the past.".format(cfg.repeat_reported_terms['limit']))
    else:
        pass
    return text


def write_trending_unigram_report(trending_df, log_dir):
    trending_log = pathlib.PurePath(log_dir, 'trending_unigrams.csv')
    trending_df.to_csv(trending_log, index=False)
    logging.info("Wrote trending unigrams report to {0}, containing {1} unigrams.".format(trending_log, len(trending_df)))
    return trending_log


def write_trending_bigram_report(trending_df, log_dir):
    trending_log = pathlib.PurePath(log_dir, 'trending_bigrams.csv')
    trending_df.to_csv(trending_log, index=False)
    logging.info("Wrote trending bigrams report to {0}, containing {1} bigrams.".format(trending_log, len(trending_df)))
    return trending_log


def write_cooccurrence_report(cooccurrence_df, log_dir):
    cooccurrence_log = pathlib.PurePath(log_dir, 'co-occurring_terms.csv')
    cooccurrence_df.to_csv(cooccurrence_log, index=False)
    logging.info("Wrote co-occurring terms report to {0}, containing {1} terms.".format(cooccurrence_log, len(cooccurrence_df)))
    return cooccurrence_log


def log_term_recommendations(text):
    term_history = {}
    columns = text.columns
    if 'word' in columns:
        terms = list(set(list(text['word'])))
    elif 'co-occurring_term' in columns:
        terms = list(set(list(text['co-occurring_term'])))
    log_file = pathlib.PurePath(cfg.log_folder, collection_name, 'term_history.json')
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
    try:
        server.login(cfg.account_to_send_emails, cfg.password_to_send_emails)
        server.sendmail(email['From'], cfg.notification_email_recipients, email.as_string())
        server.quit()
    except:
        logging.warning("Unable to authenticate the email address to send email notifications. Check the credentials in config.")


def main():
    duration = 0
    logging.info("Searching for new terms to be used alongside the existing collection criteria ({0}).".format(collection_terms))
    log_folder = pathlib.PurePath(cfg.log_folder, collection_name, str(datetime.date.today()), str(datetime.datetime.now().hour))
    try:
        os.makedirs(log_folder, exist_ok=True)
    except:
        logging.critical("Couldn't make a folder for output. Make sure you have write permissions for this directory.")
        sys.exit()
    for interval in cfg.time_interval:
        text1, text2 = find_text(interval)
        if cfg.trending_unigrams:
            trending_start = time.process_time()
            trending_df, trending_context = find_trending_words(text1, text2, stops_w_collection_terms)
            '''
            I haven't figured out what to do with trending_context yet. No idea how to report this.
            '''
            trending_stop = time.process_time()
            trending_time = trending_stop - trending_start
            logging.debug("Took {} seconds to calculate trending unigrams".format(trending_time))
            trending_df = limit_repeat_reports(trending_df)
            log_term_recommendations(trending_df)
            if trending_df.shape[0] > 0:
                trending_log = write_trending_unigram_report(trending_df, log_folder)
            elif trending_df.shape[0] == 0:
                trending_log = "No trending log"
        elif not cfg.trending_unigrams:
            logging.info("Not searching for trending unigrams.")
            trending_log = "Not searching for trending unigrams"
            trending_df = pd.DataFrame(columns=['word'])
            trending_time = 0
        duration += trending_time
        if cfg.trending_bigrams:
            trending_start = time.process_time()
            trending_bigram_df, trending_bigram_context = find_trending_bigrams(text1, text2)
            '''
            I haven't figured out what to do with trending_context yet. No idea how to report this.
            '''
            trending_stop = time.process_time()
            trending_time = trending_stop - trending_start
            logging.debug("Took {} seconds to calculate trending bigrams".format(trending_time))
            trending_bigram_df = limit_repeat_reports(trending_bigram_df)
            #log_term_recommendations(trending_bigram_df)
            if trending_bigram_df.shape[0] > 0:
                trending_bigram_log = write_trending_bigram_report(trending_bigram_df, log_folder)
            elif trending_bigram_df.shape[0] == 0:
                trending_bigram_log = "No trending log"
        elif not cfg.trending_bigrams:
            logging.info("Not searching for trending bigrams.")
            trending_bigram_log = "Not searching for trending bigrams"
            trending_bigram_df = pd.DataFrame(columns=['bigram'])
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
        if cfg.send_notification_email:
            #try:
            email_notifications(cooccurrence_log, trending_log, trending_df, cooccurrence_df)
            #except:
            #    logging.warning("Unable to send email. Are you connected to the internet?")
    return duration

running = True
while running:
    start_time = datetime.datetime.now()
    collection_terms = get_collection_terms()
    stops = build_stopwords()
    stops_w_collection_terms = stops.copy()
    stops_w_collection_terms.extend(collection_terms)
    duration = main()
    if not cfg.FlockWatch_scheduling['repeat']:
        running = False
    elif cfg.FlockWatch_scheduling['repeat']:
        now = datetime.datetime.now()
        duration = now - start_time
        resume_time = start_time + datetime.timedelta(hours=cfg.FlockWatch_scheduling['repeat_interval'].hour, minutes=cfg.FlockWatch_scheduling['repeat_interval'].minute)
        sleep_time = (resume_time - now).seconds
        if sleep_time < 0:
            logging.warning("FlockWatch takes too long to complete with your parameters for it to run as frequently as you want. FlockWatch will run again as soon as it can.\n")
        elif sleep_time > 0:
            logging.info("FlockWatch is complete. Sleeping for {0} seconds, resuming at {1}.\n".format(sleep_time, resume_time))
            time.sleep(sleep_time)

"""
Next steps:
- Add option to report context for trending words and co-occurrences
    - Context generated.
    - Need to figure out how to report context.
"""

logging.info("Process finished\n")
