1- I implemented and trained a word-level LSTM VAE on both PTB and WikiText2 datasets. Was able to get to perplexity of 120 and 150 with 10 epochs of training and only dropout regularization (possibility to reduce ppl significantly by optimizing the architecture and regularization upto SOTA language modeling results). 

2- I am able to conditionally generate text by feeding a single word to the encoder. The decoder then generates words using encoded representation of the initial word and it's last generate word. A sample generated text (100 words) on WikiText2 using initial word of "Rugby":

"  rugby 's history offensive machine battalion in canada and completed on the back as a 20 – yard victory at the rough sea line between the 150th and december 1955 . the team moved on to two championship road , tying them to pass tech back once as the fumble off to rafael djokovic ( andy on third point , ball aggregate , over 8 – 25 . <EOS> lsu won 39 points for world 2009 , surpassing dover 's longest defensive draft to 67 games . the wolfpack finished during the first season in a worcestershire @-@ torn line called  "

3- Will be moving on to the Thomson-Reuters dataset and generating text conditioned on knowledge graph on the application side of Neo Project. 

4- Want to discuss possible publication routes as I am not familiar with NLP conferences and have not delved very deeply into NLP literature. Items I want to discuss are possible problems to focus initial paper on, appropriate publication venues, benchmark datasets, required experiments, and a rough time-line.  
