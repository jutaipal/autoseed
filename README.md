totalautomodelIC is shell script based on original autoseed, it runs spacek40, with optional convert (if you want .png logos). 
It is limited to 64 bp max sequence width, and cannot count kmers beyond memory capacity (approx. 12 bp max)

autoseed_v2.2.py is Pythonized version (thanks Claude Sonnet) of the script that runs the shorter C programs localmax-motif, seedextender, motifsimilarity and genint-PWM.
it can take longer sequences as input, converting them to 2 bit encoded bitstream, and can detect longer seeds by using seedextender, which extends shorter seeds.

Please have Claude or ChatGPT read the code if you are interested in the details, they seem to be pretty good at understanding the specifics
