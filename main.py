import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import ttest_ind
from tabulate import tabulate

# Wczytanie pliku
data = pd.read_excel('thyroid.xls')
# Wyświetlenie nagłówka
#print(data.head())
# Liczność klas
#print(data['Class'].value_counts())
# Oddzielenie cech i klas
X = data.drop('Class', axis = 1)
Y = data['Class']
ranking = SelectKBest(chi2, k = 21).fit(X,Y)
# Wydzielamy cechy i przyporządkowane do nich punkty
rankingScores = pd.DataFrame(ranking.scores_)
rankingColumns = pd.DataFrame(X.columns)
# łączymy  kolumny klas i wyników
featureScores = pd.concat([rankingColumns, rankingScores], axis = 1)
featureScores.columns = ['Cecha', 'Wynik']
# Posortowany wynik od najbardziej do najmniej znaczącego
rankingSorted = featureScores.nlargest(21,'Wynik')

# Zdefiniowanie klasyfikatorów
clfs = {
    'MLPu100' : MLPClassifier(hidden_layer_sizes=(100,), solver='sgd', momentum=0, max_iter=1000, random_state=5),
    'MLPu200' : MLPClassifier(hidden_layer_sizes=(200,), solver='sgd', momentum=0, max_iter=1000, random_state=5),
    'MLPu300' : MLPClassifier(hidden_layer_sizes=(300,), solver='sgd', momentum=0, max_iter=1000, random_state=5),
    'MLPu100Moment' : MLPClassifier(hidden_layer_sizes=(100,), solver='sgd', momentum=.75, max_iter=1000, random_state=5),
    'MLPu200Moment' : MLPClassifier(hidden_layer_sizes=(200,), solver='sgd', momentum=.75, max_iter=1000, random_state=5),
    'MLPu300Moment' : MLPClassifier(hidden_layer_sizes=(300,), solver='sgd', momentum=.75, max_iter=1000, random_state=5)
}
# Krotność walidacji
n_splits = 2
# Liczba powtórzeń walidacji krzyżowej
n_repeats = 5
# Liczba wszystkich cech
n_features = 10

# # Wielokrotna walidacja krzyżowa
# rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
# scores = np.zeros((len(clfs), n_features * n_repeats * n_splits))
#
# scores_column = 0
# scores_row = 0
# for clf_name in clfs:
#     for number in range(1,n_features+1):
#         X_new = X.drop(rankingSorted['Cecha'][number::], axis = 1)
#         for train_index, test_index in rkf.split(X_new):
#             X_new_train, X_new_test = X_new.iloc[train_index], X_new.iloc[test_index]
#             Y_train, Y_test = Y[train_index], Y[test_index]
#             clf = clone(clfs[clf_name])
#             clf.fit(X_new_train, Y_train)
#             predict = clf.predict(X_new_test)
#             scores[scores_column, scores_row] = accuracy_score(Y_test, predict)
#             print(scores_row)
#             scores_row += 1
#     scores_column += 1
#     scores_row = 0
#
# np.save('results', scores)

##########################################
########## ANALIZA STATYSTYCZNA ##########
##########################################

# Wczytywanie wyników ekspermentu
scores = np.load('results.npy')

# Definiowanie analizy
alpha = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

# Listy z wartościami średniej i odchylenia
# Dla wszystkich liczb cech
features_mean = []
features_std = []

# Uzupełnienie list ^^^
for i in range(len(clfs)):
    feature_mean = []
    feature_std = []
    for j in range(n_features):
        feature_mean.append(np.mean(scores[i, j:j+10]))
        feature_std.append(np.std(scores[i, j:j+10]))
    features_mean.append(feature_mean)
    features_std.append(feature_std)

# Zdefiniowanie nazw kolumn i wierszy
headers = ["MLPu100", "MLPu200", "MLPu300", "MLPu100Moment", "MLPu200Moment", "MLPu300Moment"]
names_column = np.array([["MLPu100"], ["MLPu200"], ["MLPu300"], ["MLPu100Moment"], ["MLPu200Moment"], ["MLPu300Moment"]])

for f in range(n_features):
    print("##########################################")
    print("Dla ilości cech: %i\n" % (f+1))

    analyse_scores = np.zeros((len(clfs), n_splits*n_repeats))
    for i in range(len(clfs)):
        for j in range(n_splits*n_repeats):
            analyse_scores[i,j] = scores[i, j+f*10]

    # t-statystyka oraz p-wartość
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(analyse_scores[i], analyse_scores[j])

    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    # Przewaga
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    # Różnice statystyczne znaczące
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alpha] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    # Wynik końcowy analizy statystycznej
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("\nStatistically significantly better:\n", stat_better_table)
    print()
