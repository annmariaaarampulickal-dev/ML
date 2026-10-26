from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
train = fetch_20newsgroups(
    subset="train", remove=("headers", "footers", "quotes")
)
test = fetch_20newsgroups(
    subset="test", remove=("headers", "footers", "quotes"),
    categories=train.target_names
)

# 2. Convert text into word counts
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(train.data)
X_test = vectorizer.transform(test.data)

y_train = train.target
y_test = test.target

# 3. Multinomial Naïve Bayes
m1 = MultinomialNB()
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)

print("Multinomial NB")
print("Accuracy:", accuracy_score(y_test, p1))
print("F1 Score:", f1_score(y_test, p1, average="weighted"))

# 4. Bernoulli Naïve Bayes
m2 = BernoulliNB()
m2.fit(X_train, y_train)
p2 = m2.predict(X_test)

print("\nBernoulli NB")
print("Accuracy:", accuracy_score(y_test, p2))
print("F1 Score:", f1_score(y_test, p2, average="weighted"))