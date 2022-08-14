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
Hovewer it helps to find 2-10 actual typos in almost each article. 

To run the library, run the following command:

```bash
python parse_text.py <path>
```
where `<path>` is either path to main .tex file of the article or path to directory containing .tex files. 
