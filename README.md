totalautomodelIC is shell script based on original autoseed, it runs spacek40 (https://doi.org/10.7554/eLife.04837.035), with optional convert (if you want .png logos in addition to .svg). 
It is limited to 64 bp max sequence width, and cannot count kmers beyond memory capacity (approx. 12 bp max)

autoseed_v2.2.py is Pythonized version (thanks Claude Sonnet) of the script that runs the shorter C programs localmax-motif, seedextender, motifsimilarity and genint-PWM (all C programs with source files loaded to GitHub).
it can take longer sequences as input, converting them to 2 bit encoded bitstream, and can detect longer seeds by using seedextender, which extends shorter seeds.

included are also test files for Autoseed_v2.2.py. Using the two .seq files as input with command
./autoseed_v2.2.py 80bpshuffled.seq STARRseq_promoter_80bp.seq 5 8 100 4 100 0.35 1 0.20
should generate the .svg file in few minutes, with the enriched human promoter logos from GP5d colorectal carcinoma cells, 
the test signal sequences are derived from random sequences as described in Sahu et al. Nature Genetics, 2022.

80bak.pfm is an example background .pfm file for generation of new sequences with the -distill option
the option places motifs on sequences on background defined by .pfm, v2.2 does not try to reproduce complex
patterns such as dimer frequencies or positions along the input sequence, just average individual match frequencies.

Code is unsupported, please have Claude or ChatGPT read the code if you are interested in the details, 
they seem to be pretty good at understanding the specifics.

Relevant references for the approach using local max seeds and multinomial motif generation are:
seed: Nitta et al., eLife 2015, particularly Fig S1; https://iiif.elifesciences.org/lax/04837%2Felife-04837-fig1-figsupp1-v1.tif/full/1500,/0/default.jpg
motif generation: Jolma et al., Cell 2013, particularly Fig 1b; https://ars.els-cdn.com/content/image/1-s2.0-S0092867412014961-gr1_lrg.jpg
