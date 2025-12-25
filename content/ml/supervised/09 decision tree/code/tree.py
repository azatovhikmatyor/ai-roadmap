import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        """Calculate Gini Impurity for labels y"""
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """Find the best feature and threshold to split"""
        best_gain = -1
        split_idx, split_threshold = None, None
        parent_gini = self._gini(y)
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                # Weighted average impurity
                n = len(y)
                child_gini = (left_mask.sum() / n) * left_gini + (right_mask.sum() / n) * right_gini
                info_gain = parent_gini - child_gini

                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree"""
        num_samples = X.shape[0]
        num_labels = len(np.unique(y))

        # Stop conditions
        if (self.max_depth and depth >= self.max_depth) or num_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": True, "class": leaf_value}

        # Find best split
        feat_idx, threshold, gain = self._best_split(X, y)

        if gain <= 0:
            leaf_value = np.bincount(y).argmax()
            return {"leaf": True, "class": leaf_value}

        # Split
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf": False,
            "feature": feat_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def fit(self, X, y):
        self.tree = self._build_tree(np.array(X), np.array(y))

    def _predict_one(self, x, tree):
        if tree["leaf"]:
            return tree["class"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in np.array(X)])


# Example usage
if __name__ == "__main__":
    # Simple dataset (XOR-like)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])

    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    print("Predictions:", preds)
