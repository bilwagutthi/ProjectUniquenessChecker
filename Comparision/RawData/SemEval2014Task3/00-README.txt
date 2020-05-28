==========================================================================================
                             SEMEVAL-2014 TASK 3
		CLSTS: Cross Level Semantic Textual Similarity
A Framework for Evaluation of Semantic Similarity Between Lexical Items of Different Sizes
==========================================================================================
                     http://alt.qcri.org/semeval2014/task3/
==========================================================================================

=============
THE TASK
=============

Semantic similarity is an essential component of many applications in Natural Language Processing (NLP). This task provides an evaluation for semantic similarity across different sizes of text, which we refer to as lexical levels. Similar to the previous Semantic Textual Similarity (STS) tasks, given a pair of lexical items, a system has to automatically measure the semantic similarity. However unlike STS that focused on comparing similar-sized texts, this task evaluates the case where larger text must be compared to smaller test.


==============
LEXICAL LEVELS
==============

Lexical levels denote different sizes of text.  This Task involves five types of comparisons between four different levels:

1- Paragraph to Sentence (paragraph2sentence)
2- Sentence to Phrase (sentence2phrase)
3- Phrase to Word (phrase2word)
4- Word to Sense (word2sense)
5- Word-usage to Sense (wordusage2sense)


==================
PACKAGE CONTENTS
==================

The trial package contains the following:

00-README.txt			this file
task-3-scorer.jar		program for scoring the outputs
<level>.trial.input.txt	tab separated input files for 5 different levels
<level>.trial.output.txt	sample tab separated output files for 5 different levels
<level>.trial.gs.txt		tab separated gold standard scores for 5 different levels

=============
INPUT FORMAT
=============

The input files for the first three levels (i.e., level 1-3) consist of two tab-separated fields:

larger_side <TAB> smaller_size

For the 4th level, i.e., Word to Sense, the input format is as follows:

Word <TAB> sense_key <TAB> sense_number

where: 
sense_key is the WordNet 3.1 sense key for the corresponding sense
sense_number is of the form "word#pos#sn" (sn is the sense number according to WordNet 3.1)

The format of the level 5 (i.e., word-usage to sense) is slightly different to help distinguish which token in the context is the usage:

context_before_the_word <TAB> target_word <TAB> context_after_the_word <TAB> sense_key <TAB> sense_number

where sense_key and sense_number are the same sense representations described above (level 4).

The target_word (column 2) denotes the target usage.

=============
GOLD STANDARD
=============

The corresponding gold standard files contain human-assigned scores for each item pair (input and gold-standard files are line-aligned).  Semantic similarity is rated according to the following scale (which is a summary of the annotation instructions):

4 -- [Very Similar]
The two items have very similar meanings and the most important ideas, concepts, or actions in the larger text are represented in the smaller text;

3 -- [Somewhat Similar]
The two items share many of the same important ideas, concepts, or actions, but those expressed in the smaller text are similar but not identical to the most important in the larger text;

2 -- [Somewhat Related but NOT Similar]
The two items have dissimilar meaning, but the shared concepts, ideas, and actions in the smaller text are related (but not similar) to those of the large text;

1 -- [Slightly Related]
The two items describe dissimilar concepts, ideas and actions, but might be likely to be found together in a longer document on the same topic;

0 -- [Unrelated]
The two items do not mean the same thing and are not on the same topic. 


==============
OUTPUT FORMAT
==============

The output file is essentially similar to the gold standard. However, an optional tab-separated column can be added in case confidence score for a pair is available.

The input and output files *must* contain the same number of lines, where line X in the output corresponds to the similarity of the pair on line X in the input file.  Missing similarity scores are not allowed (i.e., every item must have a rating).


============
EVALUATION
============

Systems will be evaluated both (1) within comparison type and (2) across all comparison types. Systems that participate only in a single comparison type will be excluded from the all-comparison system ranking.  However, their inclusion in the single-comparison type setting will enable us to identify any performance gap between more general systems and specialized ones.

The system outputs and gold standard ratings will be compared in two ways, both using Krippendorff's alpha. In the first comparison, systems are evaluated according to the rating's value by calculating Krippendorff's alpha using the interval level of measurement.  For the second, we recognize that some systems may differ in their similarity measurements relative to human judges (e.g., scaling the ratings by a constant factor), but would still be able to rank the pairs correctly.  Therefore, the second comparison will calculate Krippendorff's alpha on the pair's rankings using the ordinal level of measurement.  By using the same measure for both evaluations, we hope to provide more easily comparable and interpretable scores, matching existing annotator studies.
  
To test the system outputs use the command: 

java -jar task-3-scorer.jar gold.key test.key


=============
PARTICIPATION
=============

Please visit our page in order to participate in the task:

http://alt.qcri.org/semeval2014/task3/

We invite potential participants to join our Google group:

https://groups.google.com/group/semeval-2014-clss




