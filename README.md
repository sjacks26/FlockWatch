# DOCUMENTATION IN PROGRESS

# FlockWatch

FlockWatch is a tool meant to help researchers build better data collections from social media platforms and other websites. It looks at existing datasets built around a list of collection terms, then recommends terms that the researcher might want to add to their collection criteria. FlockWatch can be run on a Linux-based server or a Mac personal computer (desktop or laptop), but it has not been tested on a Windows machine. It can be set to run every so often indefinitely, or it can be run as a one-off process.

## Getting data to FlockWatch
FlockWatch was designed to work with Twitter data collected by [STACK](https://github.com/bitslabsyr/stack). If you use STACK, FlockWatch knows how to find collection terms and data about tweets automatically based on the STACK project name.  
If you don't use STACK, you can tell FlockWatch to look for text data in a CSV; you will need to manually provide the collection terms used to collect that data in [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py#L24). With a CSV, you can use data collected from anywhere -- Facebook, Reddit, forums, even offline or digitized sources. Note that if you use a CSV, FlockWatch should only be used as a one-off process.  


### Installation and setup
1) Clone the code to your server or computer using `git clone https://github.com/sjacks26/FlockWatch.git`. You should run this command from a directory that your user has write permissions in; otherwise, you can run ServerReport as sudo.  
2) Rename `config_template.py` to `config.py`.
3) Change the parameters in [config.py](https://github.com/sjacks26/FlockWatch/blob/deploy/config_template.py) to suit your needs. See below for an explanation of these parameters.


DOCUMENTATION IN PROGRESS

main.py is the primary code for FlockWatch.  
time_analysis/trime_analysis.py is another supplemental file that looks out the output of co-occurrences over time.  
