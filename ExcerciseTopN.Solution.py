import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import operator


class Jaccard:

    def __init__(self):
        self.item_ratings_sorted = None
        self.train_set = None
        self.item_item_counts = {}
        self.item_counts = None


    def learn_model(self, train_set):
        print('Started training')
        self.train_set = train_set
        self.item_counts = self.train_set.groupby('itemID')['userID'].agg('count')

        users_grouped_items = train_set.groupby('userID')
        for user, user_data in tqdm(users_grouped_items):
            if len(user_data.index) < 3:
                continue

            picked_items = user_data.itemID.tolist()

            for i1 in range(len(picked_items)):
                item1 = picked_items[i1]

                for i2 in range(i1 + 1, len(picked_items) - 1):
                    item2 = picked_items[i2]

                    if item1 not in self.item_item_counts:
                        self.item_item_counts[item1] = dict()
                    if item2 not in self.item_item_counts[item1]:
                        self.item_item_counts[item1][item2] = 0
                    self.item_item_counts[item1][item2] += 1
                    if item2 not in self.item_item_counts:
                        self.item_item_counts[item2] = dict()
                    if item1 not in self.item_item_counts[item2]:
                        self.item_item_counts[item2][item1] = 0
                    self.item_item_counts[item2][item1] += 1
        print('Done training')

    def get_top_n_recommendations(self, test_set, top_n):
        print('Started computing recommendations')
        result = {}
        already_ranked_items_by_users = self.train_set.groupby('userID')['itemID'].apply(list)

        for userID in tqdm(test_set.userID.unique().tolist()):
            result[str(userID)] = []
            maxvalues = defaultdict(int)

            for i in already_ranked_items_by_users[userID]:
                if i not in self.item_item_counts:
                    continue

                items = self.item_item_counts[i]

                for j in items:
                    if j in already_ranked_items_by_users[userID]:
                        continue

                    if items[j] > 10:
                        d = items[j] / (self.item_counts[i] + self.item_counts[j] - items[j])
                        if d > maxvalues[j]:
                            maxvalues[j] = d

            top_list = sorted(maxvalues.items(), key=lambda kv: -kv[1])
            result[str(userID)] = [x[0] for x in top_list[:top_n]]
        print('Done computing recommendations')
        return result

    def clone(self):
        pass


train_set_path = 'resources//train_numerized_with_anon.csv'
test_set_path = 'resources//test_numerized_with_anon.csv'

train_set = pd.read_csv(train_set_path, parse_dates=[3], index_col='index')
test_set = pd.read_csv(test_set_path, parse_dates=[3], index_col='index')

users_in_train = train_set.userID.unique()

# Filter out new users from the test set
test_set = test_set[test_set.userID.isin(users_in_train)]


jaccard = Jaccard()
jaccard.learn_model(train_set)
jaccard_recs = jaccard.get_top_n_recommendations(test_set,top_n=5)
print(jaccard_recs['431'])
