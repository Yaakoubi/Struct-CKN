import numpy as np


class Sampler:
    """Implement a binary tree over changing scores.

    One can sample from proportionally to these scores if the method is RANDOM.
    Or one can find the maximum score if he method is DETERMINISTIC.

    Complexity of changing one score is O(log n). Complexity of sampling is O(log n). Complexity
    of initialization is O(n log n).
    """

    def __init__(self, score_leaves, is_determinist=False):

        self.is_determinist = is_determinist
        self.func = np.add if not self.is_determinist else np.maximum

        self.size = len(score_leaves)
        self.score_tree = [np.array(score_leaves)]
        score_level = self.score_tree[-1]
        n = score_level.shape[0]
        while n > 1:
            if n % 2 == 0:
                upper_level = self.func(score_level[::2], score_level[1::2])
                self.score_tree.append(upper_level)
            else:
                upper_level = np.empty(n // 2 + 1)
                upper_level[:-
                            1] = self.func(score_level[:-1:2], score_level[1::2])
                upper_level[-1] = score_level[-1]
                self.score_tree.append(upper_level)
            score_level = self.score_tree[-1]
            n = score_level.shape[0]

    def update(self, new_score, index):
        lower_level = self.score_tree[0]
        lower_level[index] = new_score
        for level in self.score_tree[1:]:
            # end of line, transfer
            if index + 1 == lower_level.shape[0] and (index + 1) % 2 != 0:
                level[-1] = lower_level[index]
                index //= 2
            else:
                index //= 2
                level[index] = self.func(
                    lower_level[2 * index], lower_level[2 * index + 1])
            lower_level = level

    def sample(self):
        if not self.is_determinist:
            return self.sample_random()
        else:
            return self.sample_max()

    def sample_random(self):
        score_sum = self.score_tree[-1][0]
        random_value = np.random.rand() * score_sum
        index = 0
        for level in self.score_tree[-2::-1]:
            if (index + 1) * 2 > level.shape[0]:  # end of line, transfer
                index = index * 2
            elif random_value < level[index * 2]:
                index = index * 2
            else:
                random_value -= level[index * 2]
                index = index * 2 + 1
        return index

    def sample_max(self):
        index = 0
        for level in self.score_tree[-2::-1]:
            if (index + 1) * 2 > level.shape[0]:  # end of line, transfer
                index = index * 2
            elif level[index * 2] >= level[index * 2 + 1]:
                index = index * 2
            else:
                index = index * 2 + 1
        return index

    def get_score(self, position):
        return self.score_tree[0][position]

    def get_total(self):
        return self.score_tree[-1][0]
