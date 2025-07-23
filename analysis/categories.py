
def categorize(value, categories):
    matches = [label for (lo, hi), label in categories if lo < value < hi]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise ValueError(f"No category found for value {value}")
    else:
        raise ValueError(f"Multiple categories found for value {value}: {matches}")
