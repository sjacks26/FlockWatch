###### FlockWatch was developed and tested using [Python 3.6](https://www.python.org/downloads/release/python-364/)
###### Operating System Compatibility: Windows, MacOS, Ubuntu

# FlockWatch

FlockWatch is a tool meant to help researchers build better data collections from social media platforms and other websites. It looks at existing datasets built around a list of collection terms, then recommends terms that the researcher might want to add to their collection criteria. FlockWatch creates reports of the words it recommends, and it also sends you an email every time it finishes running to let you know how many words it recommended.  

FlockWatch can be run on a Linux-based server or a Mac computer (desktop or laptop), but it has not been tested on a Windows machine. It can be set to run every so often indefinitely, or it can be run as a one-off process.

## User skills required for FlockWatch

In order to use FlockWatch, you should:
* be at least minimally comfortable with python3
* know how to edit files using vim, nano, or another command-line text editor (though you can avoid this if you are using FlockWatch on a personal computer rather than a server) 

## Getting data to FlockWatch
FlockWatch was designed to work with Twitter data collected by [STACK](https://github.com/bitslabsyr/stack). If you use STACK, FlockWatch knows how to find collection terms and data about tweets automatically based on the STACK project name.  

If you don't use STACK, you can tell FlockWatch to look for text data in a CSV. With a CSV, you can use data collected from anywhere -- Facebook, Reddit, forums, even offline or digitized sources. You will need to manually provide the collection terms used to collect that data in [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L21). Note that if you use a CSV, FlockWatch should only be used as a one-off process.  

### Installation and setup
1) Clone the code to your server or computer using `git clone https://github.com/sjacks26/FlockWatch.git`. You should run this command from a directory that your user has write permissions in; otherwise, you can run ServerReport as sudo.  
2) Install requirements   
    * If you use anaconda or miniconda, use `conda install --yes --file requirements.txt`
    * If you don't use anaconda or miniconda, use `pip install -r requirements.txt`
    * Confirm that you have NLTK's stopwords installed. If you're not sure, you can try importing the stopwords in python, or you can run `python -m nltk.downloader stopwords` to download the stopwords (and it will let you know if they're already installed).
3) Rename `config_template.py` to `config.py`.  
4) Change the parameters in [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py) to suit your needs. See below for an explanation of these parameters.  

### Setting [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py) parameters

Before you run FlockWatch, you need to set a number of parameters in the config file:
1) Tell FlockWatch whether it should [run repeatedly](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L4) in the background.  
    * If it should, tell FlockWatch [how often to run](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L5).  
2) Tell FlockWatch whether to get data [from Mongo](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L9) (if you are using STACK to collect data) or [from a CSV](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L10).
    * If Mongo:  
        a. Tell FlockWatch whether Mongo is [password-protected](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#12). If it is, give FlockWatch the Mongo [username](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L13) and [password](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L14).     
        b. Tell FlockWatch whether you want it to [ignore retweets](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L15) or include them when looking for trending or co-occurring terms.
    * If CSV:  
        a. Tell FlockWatch [where](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L18) to find the CSV. (See the [sample CSV file](https://github.com/sjacks26/FlockWatch/blob/deploy/sample_csv_file.csv) to see how to format your CSV file. Your file can contain additional columns, but it should at least contain the two columns as shown in that sample file.)  
        b. Tell FlockWatch the [name](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L19) of the column containing the information about when a message was created.  
        c. Give FlockWatch the [list of terms](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L21) used to collect the data in the CSV.
3) Tell FlockWatch what [time interval](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L15) to use when calculating trending rates. The recent block of text will include data from the interval closest to the start time (t2, by default, the current time), and the previous block of text will include data from the interval before the recent block of text (t1). You can also tell FlockWatch that you want it to use more than one interval. If you provide more than one interval, it will generate output for each interval separately.  
4) Tell FlockWatch whether you want it to [start with messages sent right now or from some previous point in time](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L25).  
5) Tell FlockWatch the [fewest number of letters](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L26) you want in any terms it suggests to you.  
6) Tell FlockWatch whether you want it to [find trending terms](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L27).  
    * Give FlockWatch the [minimum trending rate](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L25) that you're interested in. The trending rate is calculated with this formula: `(t2 - t1)/((t2 + t1)/2) * 100` where t2 is the number of times a term appears in messages in the more recent set of text and t1 is the number of times that term appears in the older set of text. The maximum trending rate (if a term appears in the more recent set of messages but not in the older messages) is 200.  
7) Tell FlockWatch whether you want it to find [terms that repeatedly appear in the same message as each collection term](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L28).  
    * Give FlockWatch the [lower threshold for co-occurrence rates](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L30) that you want it to report. Each time a given word appears in the same message as a given collection term, the co-occurrence rate for that word-collection term pair increases by 1 / (the number of messages that contain that collection term). This means that if every message that contains a particular collection term also contains a particular term, the co-occurrence rate for that word-collection term pair will be 1; if half of the messages that contain a particular collection term also contain a particular term, the co-occurrence rate for that word-collection term pair will be .5.  
9) Tell FlockWatch if you want to [stop having a term suggested after a certain number of times](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L32). (For example, you might want FlockReport to stop suggesting that you use "bomb" as a collection term after it recommends that term a couple of times, since you realize that "bomb" can mean lots of different things and thus isn't a very good collection term on its own.)  
     * If you want FlockWatch to only report terms a certain number of times, tell FlockWatch what that [limit](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L33) is.  
12) If you have a list of collection- or platform- specific [stopwords](http://www.nltk.org/book/ch02.html#stopwords_index_term) that you want it to use, tell it where to find the [file containing that list](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L38).   
13) Tell FlockWatch which [email addresses should receive an email](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L41) summarizing the FlockWatch output whenever FlockWatch generates trending and co-occurrence reports.  
    * FlockWatch will try to send this email using a Gmail account. Give FlockWatch the [Gmail account name](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L42) (without "@gmail.com") and the [password](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L43) for that account.  
15) The config parameters about context ([here](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L35) and [here](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L36)) currently don't affect output. In the future, FlockWatch will be able to provide examples of messages containing trending terms or co-occurrence pairs.   

### Running FlockWatch  

Once you have set all of the parameters in the config file, you are ready to run FlockWatch!
   
It often takes FlockWatch half an hour to 4 hours to identify trending terms and co-occurrence pairs; depending on the number of messages in each time interval and the number of collection terms, it could take more time or less time. Because of how long it takes, it's a good idea to run FlockWatch in the background using a computer that will not be doing much else while FlockWatch runs.  

When you run FlockWatch from the command line, it expects you to provide the name of the collection as a command-line argument.

Given this, the recommended way to run FlockWatch is with `python FlockWatch.py -n CollectionName &`. You should replace "CollectionName" with the name of your data collection (if you're using STACK data, this is the name of the STACK project). Including "&" at the end of this command tells the computer to run this code in the background; if you're running this on a server, that will let you do other things from the command line or close the SSH tunnel without crashing FlockWatch.

**Important!**  
FlockWatch is a _very_ CPU-intensive process. You should try to avoid running it at the same time as other CPU-intensive processes.   

### Finding and acting on output

#### Output structure
When FlockWatch runs, it creates a folder structure for output that starts with `log/` in the top folder of FlockWatch. Inside `log/`, you'll find a folder for each collection name you've used with FlockWatch. Inside of each collection's folder, you'll find another set of folders whose names correspond to each day you've run FlockWatch on that collection. Inside each of these day folders, you'll find yet another set of folders with timestamps corresponding to each time you've run FlockWatch on that day on that collection. Each of these folders will contain the trending and/or co-occurrence reports that FlockWatch generated.  

For example, if you ran FlockWatch on a collection called "test_collection" on January 1 1950 at 9:00am, you would find the reports in `FlockWatch/log/test_collection/1950-01-01/09/0/0/`.  

#### Output files
FlockWatch creates two kinds of reports: trending term reports and co-occurrence reports.  
  * Trending term reports are CSV files with four columns: a term, the number of times it appears in the older set of text, the number of times it appears in the newer set of text, and a normalized rate of change (where 200 is the maximum value). This list is sorted by rate of change; for terms that have the same rate of change, terms that appear more are higher in the list.  
  * Co-occurrence reports are CSV files with three columns: a collection term, a term that co-occurs with that collection term, and their normalized rate of co-occurrence (a co-occurrence value of 1 means that every message that contains that collection term also contains that co-occurring term). This list of co-occurrence pairs is sorted by rate of co-occurrence.  
 
#### What to do with output  
FlockWatch is meant to help you build better data collections by recommending terms related to the terms you use as collection criteria, but FlockWatch does not make any decisions about which of these related terms to use. It provides you with the lists of trending terms and co-occurrence pairs to help you make that decision.  

After FlockWatch generates output, you should look at that output and decide for yourself which of the terms in those lists you want to use. Then, you should modify your collection tool to add the terms you want to add.   

## time_analysis

The main function of FlockWatch is to provide lists of trending terms and terms that regularly appear alongside collection terms used to build datasets. To do this, it analyzes snapshots of datasets. FlockWatch is designed to repeat this snapshot analysis many times for the same dataset.   

This feature means that FlockWatch can also track which terms (if any) co-occur with a particular collection term for an extended period of time. [time_analysis.py](https://github.com/sjacks26/FlockWatch/blob/deploy/time_analysis/time_analysis.py) does this longitudinal analysis, producing line charts showing co-occurrences rates for 5 terms that co-occur with each collection term most frequently. To do this, it looks at the co-occurrence reports generated for that collection.  

#### To run time_analysis
1) First, in time_config.py, specify the [date and time](https://github.com/sjacks26/FlockWatch/blob/deploy/time_analysis/time_config.py#L3) you want FlockWatch to start looking at co-occurrence pairs for time_analysis.  
2) Next, run time_analysis with `python time_analysis/time_analysis.py -n CollectionName &`. As with FlockWatch, replace "CollectionName" with the name of your collection.  

#### time_analysis output

time_analysis creates output using a similar folder structure as the main FlockWatch function. It will create a png file for each of the collection terms in the collection you tell it to analyze.

## Future features

In the future, FlockWatch will:
1) give the user some in-context examples of the trending terms and co-occurrence pairs that it reports
2) look for trending bigrams 

## FAQs

##### What languages does FlockWatch work with?  
FlockWatch was developed based only on English-language text. A brief test with Cyrillic text indicates that FlockWatch doesn't crash with non-English text, but that doesn't necessarily mean that it generates meaningful output. If you use FlockWatch with non-English text and it provides helpful output, let me know!  

##### FlockWatch is crashing and I don't know why. Help?
If FlockWatch is crashing, the first thing to do is check the log file (based on your collection name) for an error message. For most errors, it should contain enough information about the error causing the crash to help you solve the problem. If you're stuck, create an issue and I'll try to help you out.

## Acknowledgements

This research was supported by [VOX-Pol](https://www.voxpol.eu/). The VOX-Pol Network of Excellence is funded by the European Union under the 7th Framework Programme for research, technological development, and demonstration, Grant Agreement No. 312827. 

