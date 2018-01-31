1- I implemented and trained a word-level LSTM VAE on both PTB and WikiText2 datasets. Was able to get to perplexity of 120 and 150 with 10 epochs of training and only dropout regularization (possibility to reduce ppl significantly by optimizing the architecture and regularization upto SOTA language modeling results). 

2- I am able to conditionally generate text by feeding a single word to the encoder. The decoder then generates words using encoded representation of the initial word and it's last generate word. A sample generated text (100 words) on WikiText2 using initial word of "Rugby":

"  rugby 's history offensive machine battalion in canada and completed on the back as a 20 – yard victory at the rough sea line between the 150th and december 1955 . the team moved on to two championship road , tying them to pass tech back once as the fumble off to rafael djokovic ( andy on third point , ball aggregate , over 8 – 25 . <EOS> lsu won 39 points for world 2009 , surpassing dover 's longest defensive draft to 67 games . the wolfpack finished during the first season in a worcestershire @-@ torn line called  "

3- Will be moving on to the Thomson-Reuters dataset and generating text conditioned on knowledge graph on the application side of Neo Project. 

4- Want to discuss possible publication routes as I am not familiar with NLP conferences and have not delved very deeply into NLP literature. Items I want to discuss are possible problems to focus initial paper on, appropriate publication venues, benchmark datasets, required experiments, and a rough time-line. 



# Publication strategy: talk with Jackie,
1- First publication: Shared Surface Realization Task (parse tree -> sentence)
- look at works of Yue Zhang (linearization)
- look at the work on shared surface realization task 2011 
- look at Claire Gardent's work on (grammer -> sentence)

2- Continuing research along the lines of: Conditioning on other latent factors in generatig text (i.e. controllable text generation)
- For example generating text with positive/negative opinions

3- structure of a quality NLP paper (i.e. ACL)
- motivating it very clearly on why this is important. 
- Generate language and evaluate BLEU score 
- analysis of why the system works well or not 
- an ablation study
- sampling outputs and lingusticly analyze the structure of the generated text

4- appropriate venues in order of importance for this line of work: 
ACL, EMNL, EACL, COLNG, INLG, TACL. 



### Thomson Reuters dataset on AWS S3

- make a ".aws/" directory inside your home directory and make a file called "credentials"
- Copy your S3 credentials for access to your bucket into this file (usually people make two sets of "s3-read" and "s3-write" credentials to prevent accidental overwriting)
- install awscli using pip
- there are a bunch of commands (like ls, cp, mv , etc) for accessing the S3 file system with awscli
- you can use those commands with sth like this "aws --profile=s3-read s3 ls s3://"



a sample input from train set:

price target to $18 from $16 <EOS> * etsy inc <etsy.o>: stifel raises target price to $18 from $15 <EOS> * etsy inc <etsy.o>: wedbush raises price target to $16 from $14ï¿½ï¿½ <EOS> * everbridge inc <evbg.o>: canaccord genuity raises price target to $33 from $28 <EOS> * everbridge inc

==== next word prediction with :

target target to $34 from $20 <EOS> * yelp inc <yelp.n>: barclays raises target price to $36 from $16 <EOS> * yelp inc <yelp.n>: barclays raises target target to $36 from $16 <EOS> * yelp inc <yelp.n>: credit genuity raises target target to $36 from $34 <EOS> * yelp inc



### Capital markets
- not on scale
- Differentiate research
- it's a high cost business, shell of content , formatting, etc (associates)
- writing styles of analysts. 
- Narrative science -> summarize and explain text



