# DOCUMENTATION IN PROGRESS

# FlockWatch

FlockWatch is a tool meant to help researchers build better data collections from social media platforms and other websites. It looks at existing datasets built around a list of collection terms, then recommends terms that the researcher might want to add to their collection criteria. FlockWatch can be run on a Linux-based server or a Mac personal computer (desktop or laptop), but it has not been tested on a Windows machine. It can be set to run every so often indefinitely, or it can be run as a one-off process.

## Getting data to FlockWatch
FlockWatch was designed to work with Twitter data collected by [STACK](https://github.com/bitslabsyr/stack). If you use STACK, FlockWatch knows how to find collection terms and data about tweets automatically based on the STACK project name.  

If you don't use STACK, you can tell FlockWatch to look for text data in a CSV. With a CSV, you can use data collected from anywhere -- Facebook, Reddit, forums, even offline or digitized sources. You will need to manually provide the collection terms used to collect that data in [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L24). Note that if you use a CSV, FlockWatch should only be used as a one-off process.  


### Installation and setup
1) Clone the code to your server or computer using `git clone https://github.com/sjacks26/FlockWatch.git`. You should run this command from a directory that your user has write permissions in; otherwise, you can run ServerReport as sudo.  
2) Rename `config_template.py` to `config.py`.
3) Change the parameters in [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py) to suit your needs. See below for an explanation of these parameters.

DOCUMENTATION IN PROGRESS

### Running FlockWatch

### time_analysis

## Requirements

FlockWatch was developed and tested using:  
* [Python 3.6](https://www.python.org/downloads/release/python-364/)  
* [nltk 3.2.5](https://pypi.org/project/nltk/3.2.5/) (including the English stopwords list, which is [installed separately](https://stackoverflow.com/questions/41610543/corpora-stopwords-not-found-when-import-nltk-library))  
* [pandas 0.22.0](https://pypi.org/project/pandas/0.22.0/)  
* [scipy 1.0.0](https://pypi.org/project/scipy/1.0.0/)  
* [pymongo 3.4.0](https://pypi.org/project/pymongo/3.4.0/)  
* [matplotlib 2.1.2](https://pypi.org/project/matplotlib/2.1.2/)  

FlockWatch.py is the primary code for FlockWatch.  
time_analysis/time_analysis.py is another supplemental file that looks out the output of co-occurrences over time.  
