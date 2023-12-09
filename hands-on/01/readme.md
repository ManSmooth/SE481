# 15-23 (`h-01.py`)
1. Load csv using pandas from Kaggle file
2. Load list of database names (`functions.parse_db`)
    1. Get request content from database engine ranking site
    2. Use bs4 to parse HTML content into Python object
    3. Extract names
    4. Choose only top 10
    5. Transform into each name into list of tokens
3. Transform dataframe from csv (`functions.transformation_pipe`)
    1. Clean `job_description` column `functions.extract_description`
    2. Tokenize `job_description` column `functions.tokenize`
4. Create new dataframe from from databases
5. For each database do in dataframe
    1. Find rows that match the database tokens eg. `["oracle"]`
    2. Sum all matches
    3. Result is the amount of rows that matches given database name.
    4. Result goes into occurance
6. Find % by dividing previous column by amount of rows
7. Same as `5.` but also with Python
8. Find relative % of `7.` and `5.` meaning, percentage of rows that matches database that also mentions python to ones with only database mentioned.

This one is for statistical analysis purposes.

# 25-26 (`h-02.py`)
1. Define Programming Languages
2. Repeat `1. - 3.` from `15-23`
3. Combine databases and programming languages to a list of terms
4. Create a document term matrix
    1. For each row
        1. For each keyword
            1. If all token matches the number is 1
            2. Otherwise it is 0
    2. Set column names according to the terms list

These can be used to easily query based on terms.

# 35-36 (`h-03.py`)
1. Define 2 strings
2. Using nltk package we download the definitions of stopwords and punkt tokenizer
3. Tokenize both strings
4. For both list of tokens choose only tokens that are longer than 2 characters
5. Remove stop words from both list of tokens
6. Using porter stemming algorithm, we stem each token ("developing", "developer" -> "develop"), and remove duplicates

These can be used to dynamically construct keywords or indexes or terms to process documents without knowing what they're about beforehand.
