import csv
import re
from collections import Counter
import time
import nltk
from nltk import bigrams, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

negation_words = [
  'no',
  'not',
  'none',
  'never',
  'neither',
  'nobody',
  'nothing',
  'nowhere',
  'seldom',
  'scarcely',
  'hardly',
  'barel',
  'cannot'
]


with open('external_resources/positive-words.txt', 'r') as f:
    positive_words = set(f.read().lower().strip('\n').split('\n'))

with open('external_resources/negative-words.txt', 'r') as f:
    negative_words = set(f.read().lower().strip('\n').split('\n'))


new_negative_words = set()
new_positive_words = set()
for negation_word in negation_words:
  new_negative_words = new_negative_words.union(set([f'{negation_word}JOINEDKEYWORD{positive_word}' for positive_word in positive_words]))
  new_positive_words = new_positive_words.union(set([f'{negation_word}JOINEDKEYWORD{negative_word}' for negative_word in negative_words]))

positive_words = positive_words.union(new_positive_words)
negative_words = negative_words.union(new_negative_words)

print(len(positive_words), len(negative_words))


def get_helpfulness(entry):
  try:
    value = int(entry['HelpfulnessNumerator']) / int(entry['HelpfulnessDenominator'])
  except:
    value = 0
  return value

def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['pos'], scores['neg'], scores['compound']

def get_positive_text(entry):
    text = entry['Text'].lower()
    for word in negation_words:
        text = re.sub(fr'\b{word}\s(\w+)', fr'{word}JOINEDKEYWORD\1', text)

    # Tokenize and count words as before
    split_entry = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    word_counters = Counter(split_entry)
    num_neg = sum(word_counters[k] for k in negative_words.intersection(word_counters))
    num_pos = sum(word_counters[k] for k in positive_words.intersection(word_counters))
    ratio = num_pos / (num_neg + num_pos) if num_neg + num_pos else 0.5  # Neutral if no sentiment words found

    # Get VADER sentiment scores for the text
    vader_pos, vader_neg, vader_compound = get_vader_sentiment(text)

    # Return both word-based and VADER-based features
    return {
        'NumNegativeText': num_neg,
        'NumPositiveText': num_pos,
        'PercPositiveText': ratio,
        'VADER_PositiveText': vader_pos,
        'VADER_NegativeText': vader_neg,
        'VADER_CompoundText': vader_compound
    }


def get_positive_summary(entry):
    text = entry['Summary'].lower()
    for word in negation_words:
        text = re.sub(fr'\b{word}\s(\w+)', fr'{word}JOINEDKEYWORD\1', text)

    # Tokenize and count words as before
    split_entry = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    word_counters = Counter(split_entry)
    num_neg = sum(word_counters[k] for k in negative_words.intersection(word_counters))
    num_pos = sum(word_counters[k] for k in positive_words.intersection(word_counters))
    ratio = num_pos / (num_neg + num_pos) if num_neg + num_pos else 0.5

    # Get VADER sentiment scores for the summary
    vader_pos, vader_neg, vader_compound = get_vader_sentiment(text)

    # Return both word-based and VADER-based features
    return {
        'NumNegativeSummary': num_neg,
        'NumPositiveSummary': num_pos,
        'PercPositiveSummary': ratio,
        'VADER_PositiveSummary': vader_pos,
        'VADER_NegativeSummary': vader_neg,
        'VADER_CompoundSummary': vader_compound
    }



def get_review_counts(entry, product_reviews, user_reviews):
  return {
    'NumProductReviews': product_reviews.get(entry['ProductId'], 1),
    'NumUserReviews': user_reviews.get(entry['UserId'], 1)
  }


print('Setting Up data-wide indexes')
product_reviews = Counter()
user_reviews = Counter()
min_time = {}
with open('train.csv', 'r') as f:
  reader = csv.reader(f)
  headers = next(reader)
  for i, line in enumerate(reader):
    entry = dict(zip(headers, line))
    product_id = entry['ProductId']
    product_reviews[product_id] += 1
    user_reviews[entry['UserId']] += 1
    if product_id not in min_time:
      min_time[product_id] = int(entry['Time'])
    else:
      min_time[product_id] = min(int(entry['Time']), min_time[product_id])


start = time.time()
text_values = []
summary_values = []
with open('train.csv', 'r') as f, open('train.modded.csv', 'w') as out_f:
  reader = csv.reader(f)
  out = csv.writer(out_f)
  headers = next(reader)
  modded_headers = headers + ['Helpfulness', 'NumNegativeText', 'NumPositiveText', 'PercPositiveText', 'NumNegativeSummary', 'NumPositiveSummary', 'PercPositiveSummary', 'NumProductReviews', 'NumUserReviews', 'TimeFromFirst']
  out.writerows([modded_headers])
  for i, line in enumerate(reader):
    if i % 10_000 == 0:
       print(f'PROCESSED {i} LINES ({round(time.time() - start, 2)}s)')
    entry = dict(zip(headers, line))
    entry['Helpfulness'] = get_helpfulness(entry)
    entry.update(get_positive_text(entry))
    entry.update(get_positive_summary(entry))
    entry.update(get_review_counts(entry, product_reviews, user_reviews))
    entry['TimeFromFirst'] = str(int(entry['Time']) - min_time[entry['ProductId']])
    if entry['PercPositiveText'] != 'N/A':
      text_values += [entry['PercPositiveText']]
    if entry['PercPositiveSummary'] != 'N/A':
      summary_values += [entry['PercPositiveSummary']]
    _ = out.writerows([[entry[k] for k in modded_headers]])


print('Average Summary Values --> ', sum(summary_values) / len(summary_values))
print('Average Text Values --> ', sum(text_values) / len(text_values))



