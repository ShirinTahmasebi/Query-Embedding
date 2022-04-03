from collections import defaultdict
import re


class QueryPattern:

    def __init__(self, key: int, pattern: str, similar_groups_keys=None):
        if similar_groups_keys is None:
            similar_groups_keys = []
        self.key = key
        self.pattern = pattern
        self.similar_groups_keys = similar_groups_keys

        if key in self.similar_groups_keys:
            self.similar_groups_keys.remove(key)


class QueryPatterns:

    def __init__(self):
        self.patterns = defaultdict()

    def add_item(self, item: QueryPattern):
        self.patterns[item.key] = item

    def get_matched_pattern_key(self, input: str):
        for key, item in self.patterns.items():
            if re.match(item.pattern, input):
                return key
        return -1

    def get_similar_patterns(self, pattern_key: int):
        return self.patterns[pattern_key].similar_groups_keys
