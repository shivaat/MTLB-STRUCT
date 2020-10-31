# README #

This is the README file from the PARSEME verbal multiword expressions (VMWEs) 
corpus for Irish, edition 1.2.

The raw corpus is not released in the present directory, but can be downloaded from a [dedicated page](https://gitlab.com/parseme/corpora/-/wikis/Raw-corpora-for-the-PARSEME-1.2-shared-task)

## Annotated Corpus ##

All annotated data (1,700 sentences) comes from the Universal Dependencies v2.5
([Irish Dependency Treebank](https://universaldependencies.org/treebanks/ga_idt/index.html)). 
The genre is news, fiction, web, and legal. Most columns were annotated manually 
before being converted to UD format, with some manual corrections made. The 
exception is the XPOS column, the tags of which were assigned by a program, with
some manual corrections, but not a full manual verification. The data are in the
.cupt format. Here is detailed information 
about some columns:

*  For most columns, including the tagset used: see the treebank documentation 
linked above.
*  PARSEME:MWE (column 11): Manually annotated by a single annotator per file. 
  *  The following VMWE categories are annotated: VID, LVC.full, LVC.cause, 
VPC.full, VPC.semi, IAV, IRV.

## Unannotated Raw Corpus ##

The unannotated raw corpus, consisting of 1,379,824 dependency parsed trees in 
CoNLL-U format, is a compilation of various sources of raw text data, which 
have been automatically tokenised, POS-tagged, lemmatised, morphologically 
analysed and dependency parsed with the UDPipe tool, using the 
[irish-idt-ud-2.5-191206.udpipe model](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131). 
The tagset used was the same as that of the annotated corpus. The raw text was 
compiled from the following sources:

* Citizens Information: 10,297 sentences crawled from the Citizen’s Information
Ireland website (https://www.citizensinformation.ie/ga/)
* EU Bookshop: 133,363 sentences from the EU bookshop, accessed from the Opus 
website (http://opus.nlpl.eu/EUbookshop.php)
* Paracrawl: 782,769 sentences from the Irish side of the Paracrawl data, 
accessed from the Opus website (http://opus.nlpl.eu/ParaCrawl.php)
* Tatoeba: 1,894 translated sentences from the Tatoeba corpus, accessed 
from the Opus website (http://opus.nlpl.eu/Tatoeba.php)
* Vicipéid: 302,838 sentences from the Irish Wikipedia text dump 
(https://dumps.wikimedia.org/gawiki/20200220/)

As the data was automatically compiled and processed, the quality cannot be assured. 
Text from EU bookshop, Paracrawl and Tatoeba was translated from the original source.
The genre is a mixture of general, legal, and news. Some automatic cleaning was performed, but
noisy text, including boilerplate text may still be included.

## Licences ##

The UD Irish Dependency Treebank is licensed under Creative Commons Share-Alike
 3.0 licence CC-BY-SA 3.0 The annotated VMWE data is licensed under Creative 
 Commons 4.0 licence CC-BY 4.0

UD Pipe is a free software distributed under the Mozilla Public License 2.0 and
the linguistic models are free for non-commercial use and distributed under the 
CC BY-NC-SA license 

Text from the Citizen’s Information contains Irish Public Sector Data licensed 
under a Creative Commons Attribution 4.0 International (CC BY 4.0) licence.

Text from ParaCrawl is under the Creative Commons CC0 Licence (“no rights 
reserved”).

Text from the EU bookshop is available for free download from the Opus website 
for open-source data. No licence was specified with the data.

Text from Tatoeba corpus is available under the CC–BY 2.0 FR licence.

Vicipéid (Irish Wikipedia) data is licensed under the GNU Free Documentation 
License (GFDL) and the Creative Commons Attribution-Share-Alike 3.0 License.

## Authors ##

Abigail Walsh, Teresa Lynn, Jennifer Foster

## Contact ##

abigail.walsh@adaptcentre.ie
