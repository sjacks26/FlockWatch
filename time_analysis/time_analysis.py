"""
This file is used to look at what terms are reported as trending and/or co-occurrences over time.
"""
import pandas as pd
import os
import logging
import sys
from dateutil.parser import parse
import argparse
import matplotlib
import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time_config as cfg

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True, help="The name of the data collection being FlockWatched. If your data comes from STACK, this should be the STACK project name.")
args = vars(ap.parse_args())

collection_name = args['name']

logging.basicConfig(filename=cfg.time_analysis_log_file,filemode='a+',level=logging.DEBUG, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")

logging.info("Analyzing terms that have been reported as trending or co-occuring since".format(str(cfg.start_date)))
log_dir = os.path.join(cfg.log_folder, collection_name)
if not os.path.exists(log_dir):
    logging.warning("No reports exist for {}. Exiting.".format(collection_name))
    sys.exit()

log_dir_contents = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir,f))]
reports_in_range = []
for f in log_dir_contents:
    log_date = parse(os.path.basename(f))
    if type(cfg.start_date) is datetime.datetime:
        if log_date >= cfg.start_date:
            reports_in_range.append(f)
    elif type(cfg.start_date) is datetime.timedelta:
        start_date = datetime.datetime.today().date() - cfg.start_date
        if log_date >= start_date:
            reports_in_range.append(f)
logging.info("Found {} sets of results.".format(len(reports_in_range)))

co_occurrence_reports = []
for f in reports_in_range:
    for path, subdir, files in os.walk(f):
        co_occurrence_files = [os.path.join(path, f) for f in files if "co-occurring" in f]
        co_occurrence_reports.extend(co_occurrence_files)
logging.info("Analyzing {0} co-occurrence reports.".format(len(co_occurrence_reports)))


co_occurrence_over_time = pd.DataFrame()
for result in co_occurrence_reports:
    date = result.split('/')[3:5]
    date = ' '.join(date)
    result_file_info = pd.read_csv(result) #, index_col='collection_term')
    result_file_info.set_index(['collection_term', 'co-occurring_term'], inplace=True)
    #result_file_info.index = list(zip(result_file_info['collection_term'], result_file_info['co-occurring_term']))
    result_file_info = result_file_info['rate']
    co_occurrence_over_time[date] = result_file_info

co_occurrence_terms = list(set(co_occurrence_over_time.index.levels[0]))

timestamp = datetime.datetime.now()
figure_folder = os.path.join('figures', collection_name, str(timestamp.date()), str(timestamp.time()))
os.makedirs(figure_folder, exist_ok=True)

for term in co_occurrence_terms:
    plot_df = co_occurrence_over_time.loc[term]
    plot_df = plot_df.transpose().sort_index()
    plot_df.sort_values(by=list(plot_df.columns), ascending=False, inplace=True)
    plot_dates = []
    plot_dates = set(plot_dates)
    plot_df = plot_df.iloc[:,:5]
    plot_df.sort_index(inplace=True)
    co_occurrence_plot = plt.figure(figsize=(10, 4))
    plot1 = co_occurrence_plot.add_subplot(111)
    for cos in plot_df.columns.values:
        plot1.plot_date(plot_df.index, plot_df[cos], fmt='-')
    #plot1 = plot_df.plot(ylim=(0, 1))
    plot1.set_ylim(0,1)
    co_occurrence_plot.autofmt_xdate()
    co_occurrence_plot.legend()
    fig_name = term+'-co-occurrence-plot'
    fig_name = os.path.join(figure_folder, fig_name)
    plt.savefig(fig_name)
    #plt.show()
    plt.close('all')

logging.info("Plots of top co-occurring terms created at {}.\nProcess complete.".format(figure_folder))