from typing import List
from collections import Counter

def count_adjecent(ids: List[int]) -> Counter:
    counter = Counter(zip(ids, ids[1:]))
    return counter

def merge_ids(
    orignal_ids: List[int], 
    pair: tuple[int, int], 
    id:int) -> List[int]:
    new_ids = []
    i = 0
    while i < len(orignal_ids)-1:
        last_match_flag  = False
        if orignal_ids[i] == pair[0] and orignal_ids[i+1] == pair[1]:
            new_ids.append(id)
            i += 2
            last_match_flag = True
        else:
            new_ids.append(orignal_ids[i])
            i += 1
    if not last_match_flag:
        new_ids.append(orignal_ids[-1])
    return new_ids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)
def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s
if __name__ == "__main__":
    ids = [1, 2, 3, 4, 5]
    print(count_adjecent(ids))
    print(merge_ids(ids, (4,5), 6))