import csv
import re
from collections import Counter
from numpy import std
import time


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


def get_positive_text(entry):
  text = entry['Text'].lower()
  for word in negation_words:
    text = re.sub(fr'\b{word}\s(\w+)', fr'{word}JOINEDKEYWORD\1', text)
  split_entry = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
  word_counters = Counter(split_entry)
  num_neg = sum([word_counters[k] for k in negative_words.intersection(word_counters)])
  num_pos = sum([word_counters[k] for k in positive_words.intersection(word_counters)])
  ratio = num_pos / (num_neg + num_pos) if num_neg + num_pos else 0.6805348032192947
  return {
    'NumNegativeText': num_neg,
    'NumPositiveText': num_pos,
    'PercPositiveText': ratio,
    'TextLength': sum(list(word_counters.values())),
    'ExclamationCount': entry['Text'].count('!')
  }


def get_positive_summary(entry):
  text = entry['Summary'].lower()
  for word in negation_words:
    text = re.sub(fr'\b{word}\s(\w+)', fr'{word}JOINEDKEYWORD\1', text)
  split_entry = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
  word_counters = Counter(split_entry)
  num_neg = sum([word_counters[k] for k in negative_words.intersection(word_counters)])
  num_pos = sum([word_counters[k] for k in positive_words.intersection(word_counters)])
  ratio = num_pos / (num_neg + num_pos) if num_neg + num_pos else 0.7390373900777611
  return {
    'NumNegativeSummary': num_neg,
    'NumPositiveSummary': num_pos,
    'PercPositiveSummary': ratio
  }



def get_review_counts(entry, product_reviews, user_reviews, average_user_rating):
  return {
    'NumProductReviews': product_reviews.get(entry['ProductId'], 1),
    'NumUserReviews': user_reviews.get(entry['UserId'], 1),
    'AverageReview': average_user_rating.get(entry['UserId'], (4.201582078507381, 0.7166681401878774))[0],
    'StdReview': average_user_rating.get(entry['UserId'], (4.201582078507381, 0.7166681401878774))[1],
  }


print('Setting Up data-wide indexes')
product_reviews = Counter()
user_reviews = Counter()
min_time = {}
average_user_rating = {}
with open('train.csv', 'r') as f:
  reader = csv.reader(f)
  headers = next(reader)
  for i, line in enumerate(reader):
    entry = dict(zip(headers, line))
    product_id, user_id = entry['ProductId'], entry['UserId']
    product_reviews[product_id] += 1
    user_reviews[user_id] += 1
    if product_id not in min_time:
      min_time[product_id] = int(entry['Time'])
    else:
      min_time[product_id] = min(int(entry['Time']), min_time[product_id])
    if user_id not in average_user_rating and entry['Score']:
      average_user_rating[user_id] = []
    if entry['Score']:
      average_user_rating[user_id] += [int(float(entry['Score']))]


for user_id, scores in average_user_rating.items():
  average_user_rating[user_id] = (sum(scores) / len(scores), std(scores))




start = time.time()
text_values = []
summary_values = []
with open('train.csv', 'r') as f, open('train.modded.csv', 'w') as out_f:
  reader = csv.reader(f)
  out = csv.writer(out_f)
  headers = next(reader)
  modded_headers = headers + ['Helpfulness', 'NumNegativeText', 'NumPositiveText', 'PercPositiveText', 'NumNegativeSummary', 'NumPositiveSummary', 'PercPositiveSummary', 'NumProductReviews', 'NumUserReviews', 'TimeFromFirst', 'TextLength', 'ExclamationCount', 'AverageReview', 'StdReview']
  out.writerows([modded_headers])
  for i, line in enumerate(reader):
    if i % 10_000 == 0:
       print(f'PROCESSED {i} LINES ({round(time.time() - start, 2)}s)')
    entry = dict(zip(headers, line))
    entry['Helpfulness'] = get_helpfulness(entry)
    entry.update(get_positive_text(entry))
    entry.update(get_positive_summary(entry))
    entry.update(get_review_counts(entry, product_reviews, user_reviews, average_user_rating))
    entry['TimeFromFirst'] = str(int(entry['Time']) - min_time[entry['ProductId']])
    if entry['PercPositiveText'] != 'N/A':
      text_values += [entry['PercPositiveText']]
    if entry['PercPositiveSummary'] != 'N/A':
      summary_values += [entry['PercPositiveSummary']]
    _ = out.writerows([[entry[k] for k in modded_headers]])


print('Average Summary Values --> ', sum(summary_values) / len(summary_values))
print('Average Text Values --> ', sum(text_values) / len(text_values))



