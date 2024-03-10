Complete the Flask application

-   Allow the user to search from Elasticsearch and the manually created index (using different route)
-   Integrate PageRank to the search results from the manually created index.
-   Combine the BM25 scores with PageRank
-   Add an html \<b\> .. \<\/b\> tag to the query term and show only two or three sentences surrounding the query term.
-   Discuss about how this new mix of scores makes finding things better or worse.

## Integrate PageRank to the search results from the manually created index.

Included Page rank into Manual Indexer

````python
def run_indexer(self):
        documents = []
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".json"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                documents.append(j)
        self.documents = pd.DataFrame.from_dict(documents)
        tfidf_vectorizor = TfidfVectorizer(
            preprocessor=custom_preprocessor, stop_words=stopwords.words("english")
        )
        tfidf_vectorizor.fit(
            self.documents.apply(lambda s: " ".join(s[["title", "text"]]), axis=1)
        )
        self.bm25 = BM25(tfidf_vectorizor)
        self.bm25.fit(
            self.documents.apply(lambda s: " ".join(s[["title", "text"]]), axis=1)
        )
        ########
        # HERE #
        ########
        self.pr = PR(0.8)
        self.pr.pr_calc()
        with open(self.stored_file, "wb") as f:
            pickle.dump(self.__dict__, f)
````

## Combine the BM25 scores with PageRank
Both scores have been scaled to normalize both. (though I don't know if it's the right call)
````python
scores = minmax_scale(self.bm25.transform([query]))
    df = pd.DataFrame(scores, columns=["score"])
    result_df = self.documents.join(df)
    result_df["score"] = result_df.apply(
        lambda x: self.pr.pr_result[x["url"]] * x["score"], axis=1
    )
````
Reused the fitted BM25 model to also score elastic search pagerank. The 0.01 is just a magic number and more of a safety net in case there's inconsistency but I don't think there are.
````python
scores = minmax_scale(app.manual_indexer.bm25.transform([query_term]))
df = pd.DataFrame(scores, columns=["score"])
bm25_scores = app.manual_indexer.documents.join(df).set_index("url")["score"]
results_df = pd.DataFrame(
    [
        [
            hit["_source"]["title"],
            hit["_source"]["url"],
            hit["_source"]["text"][:100],
            (
                hit["_score"] * bm25_scores[hit["_source"]["url"]]
                if hit["_source"]["url"] in bm25_scores
                else 0.01
            ),
        ]
        for hit in results["hits"]["hits"]
    ],
    columns=["title", "url", "text", "score"],
).sort_values("score", ascending=False)
````

## Add an html \<b\> .. \<\/b\> tag to the query term and show only two or three sentences surrounding the query term
As you can see, the design is very human.
````python
result_df["text_highlight"] = result_df["text"].apply(
    lambda x: [
        f"...{x[span[0] - 24: span[0]]}<b>{x[span[0] : span[1]]}</b>{x[span[1]: span[1] + 24]}..."
        for span in [
            word_span
            for word_spans in [
                [
                    m.span()
                    for m in re.finditer(rf"\b{q_word}\b", x, re.IGNORECASE)
                ]
                for q_word in query.split()
            ]
            for word_span in word_spans
        ]
    ]
)
````
1. Accounting for multiple query, we use regex to search for the span of occurances for each term.
2. We then flatten it.
3. Find characters around the each span and format them a little.

Better way to do this would be to find all words matching any of the terms, and then have a sliding window to determine the highest density of match and choose that window to display.

## Discuss about how this new mix of scores makes finding things better or worse.
I think for corpus as small as this, PageRank hurts more than help, especially if you give them equal weight. Though I think it would work better if we have more data for both BM25 and PageRank to give better heuristics.
