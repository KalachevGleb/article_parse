# Library for parsing TeX sources of scientific articles

## Modules

- Tex document parser (file texdocument.py):
    - Parses a TeX document expanding user-defined macros and returns a TexDocument object.
    - The TexDocument object contains the structure of parsed document and can be used to
      extract information from the document.
- Text parser (file parse_text.py):
    - Parses sentenses found in text and returns parse trees.
    - Reports errors and warnings found during parsing.
    - Current parsing quality is insufficient for deep semantic analysis, 
      but can be used to detect some types of errors and typos.
    - Currently, supports only English language.

## Possible usage

Can be used to find errors in TeX sources. Currently, it performs following checks:
- Use of words that are rare in scientific articles (dictionary of frequent words collected from 10000 random arxiv articles)
- Some grammatical errors (e.g. wrong verb form used with plural/singular form of nouns, missed commas, etc.)
- Some formatting errors in formulas 
  (e.g. mismatching parentheses or their sizes, small parentheses around big expressions, etc.)

**Note:** > 50% of reported errors may be false positives due to incorrect parsing. 
Hovewer, it helps to find 2-10 actual typos in almost each article. 

### Running parser

To run the parser, use the following command:

```bash
python parse_text.py <path>
```
where `<path>` is either path to main .tex file of the article or path to directory containing .tex files.

### Output

1. First part of output is list of uncommon words found in the article.
Currently, it doesn't report positions of these words in source files, so if some of these words
contains a typo, they can be found by searching for these words in a text editor.
2. The list of warnings and errors found during parsing.
Since currently preprocessor doesn't save locations in source files, 
for each found problem it reports the sentence or a part of the sentence where the problem was found.
In this case the sentence is also can be found by searching its part in a text editor.
3. If problem was found in formula, the formula with found promlem is printed (with expanded macros). 
4. Following files are generated for debugging porposes:
   - **preprocessed.tex**: preprocessed TeX source with expanded user-defined macros and removed comments.
   - **parsed.txt**: contains parse trees of all sentences that were parsed (hovewer, some of them maybe parsed incorrectly).
   - **failed.txt**: contains sentences that were not entirely parsed (parts of sentence represented as separate syntax trees).
   - **parse_trees.txt**: contains parse trees of all sentences (union of parsed.txt and failed.txt).

### Existing problems

1. Words containing two parts separated by dash (e.g. well-known) are 
usually reported as rare even if they are not actually rare.
2. If sentence is not parsed correctly, errors in it may not be reported 
   (usually, < 50% sentences are fully parsed). 
3. Some macros are not expanded and are interpreted as parts of the sentence.
4. Figures and tables, including their captions, are excluded from parsing.
5. '.' after abreviation can be treated as the end of sentence.
