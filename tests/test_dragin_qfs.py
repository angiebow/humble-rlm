from dragin_rlm.qfs import format_query, select_query_tokens


def test_select_query_tokens_picks_highest_attention_and_restores_order():
    tokens = ["In", " 1903", ",", " he", " secured", " a", " job"]
    attn = [0.05, 0.6, 0.01, 0.1, 0.55, 0.02, 0.3]
    selected = select_query_tokens(tokens, attn, top_n=3)
    # top-3 by attention: " 1903" (0.6), " secured" (0.55), " job" (0.3)
    # restored to original left-to-right order:
    assert selected == [" 1903", " secured", " job"]


def test_select_query_tokens_respects_top_n():
    tokens = ["a", "b", "c", "d"]
    attn = [0.1, 0.9, 0.5, 0.2]
    assert select_query_tokens(tokens, attn, top_n=1) == ["b"]
    assert select_query_tokens(tokens, attn, top_n=100) == tokens  # capped at len


def test_select_query_tokens_handles_empty_input():
    assert select_query_tokens([], [], top_n=5) == []
    assert select_query_tokens(["a"], ["not-a-row-of-matching-length"], top_n=0) == []


def test_format_query_joins_selected_tokens():
    tokens = ["Einstein", " secured", " a", " job", " at", " Zurich"]
    attn = [0.9, 0.8, 0.05, 0.1, 0.05, 0.7]
    q = format_query(tokens, attn, top_n=3)
    assert q == "Einstein secured Zurich"


def test_format_query_rebuilds_whole_words_from_subword_tokens():
    # "Corliss" is split across tokens; the query must carry the whole word.
    from dragin_rlm.qfs import format_query

    tokens = ["Who", " is", " Cor", "liss", " Archer", "?"]
    attn = [0.0, 0.0, 0.9, 0.8, 0.1, 0.0]
    assert format_query(tokens, attn, top_n=2) == "Corliss"


def test_format_query_drops_thinking_tags():
    from dragin_rlm.qfs import format_query

    tokens = ["<think>", " the", " </think>", " Corliss"]
    assert format_query(tokens, [1.0] * 4, top_n=4) == "the Corliss"


def test_format_query_without_leading_space_markers_uses_tokens_as_is():
    from dragin_rlm.qfs import format_query

    assert format_query(["Cor", "liss"], [0.5, 0.9], top_n=1) == "liss"
