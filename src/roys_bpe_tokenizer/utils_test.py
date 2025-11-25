from roys_bpe_tokenizer.utils import count_adjecent

def test_count_adjecent():
    ids = [1, 2, 3, 4, 5]
    expected = {
        (1, 2): 1,
        (2, 3): 1,
        (3, 4): 1,
        (4, 5): 1,
    }
    assert count_adjecent(ids) == expected