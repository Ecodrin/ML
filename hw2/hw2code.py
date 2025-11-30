import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    thresholds = []
    ginis = []
    threshold_best = -1
    gini_best = -1

    feature = np.array(feature_vector)
    target = np.array(target_vector)

    sorted_feature_arg = np.argsort(feature)
    feature = feature[sorted_feature_arg]
    target = target[sorted_feature_arg]

    features_diff = np.diff(feature)
    maybe_thresholds_indices = np.where(features_diff > 0)[0]
    if len(maybe_thresholds_indices) == 0:
        return None, None, None, None

    thresholds = (feature[maybe_thresholds_indices] + feature[maybe_thresholds_indices + 1]) / 2.0

    cum_ones = np.cumsum(target)
    number_class1 = cum_ones[-1]
    Rn = len(target_vector)

    Rl = maybe_thresholds_indices + 1
    Rr = Rn - Rl

    number_class1_left = cum_ones[maybe_thresholds_indices] 
    number_class1_right = number_class1 - number_class1_left    
    pl = number_class1_left / Rl
    pr = number_class1_right / Rr

    gini_left = 1 - pl ** 2 - (1 - pl) ** 2
    gini_right = 1 - pr ** 2 - (1 - pr) ** 2

    ginis = -(Rl / Rn * gini_left + Rr / Rn * gini_right)

    best_gini_ind = np.argmax(ginis)
    gini_best = ginis[best_gini_ind]
    threshold_best = thresholds[best_gini_ind]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {"types": self._feature_types}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and self._max_depth <= depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y,).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y,).most_common(1)[0][0]
            return


        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is not None and (gini_best is None or gini > gini_best):
                split = feature_vector < threshold

                Rl = np.sum(split)
                Rr = len(sub_y) - Rl

                if self._min_samples_leaf is not None and (self._min_samples_leaf > Rl or self._min_samples_leaf > Rr):
                    continue

                feature_best = feature
                gini_best = gini

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth=depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth=depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature_split = node['feature_split']
        if self._feature_types[feature_split] == 'real':
            threshold = node['threshold']
            if x[feature_split] < threshold:
                return self._predict_node(x, node['left_child']) 
            else:
                return self._predict_node(x, node['right_child'])
        else:
            category = x[feature_split]
            cat_split = node['categories_split']
            if category in cat_split:
                return self._predict_node(x, node['left_child']) 
            else:
                return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
