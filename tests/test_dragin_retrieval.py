from dragin_rlm.retrieval import bm25_rank, slice_context


DOC = (
    "The Androscoggin Bank Colisee is a multi purpose arena in Lewiston Maine "
    "that opened in nineteen fifty eight. It has a seating capacity of three "
    "thousand six hundred seventy seven people.\n\n"
    "The Kansas City metropolitan area grew steadily through the twentieth "
    "century, driven by manufacturing and later by logistics and healthcare.\n\n"
    "University of Kansas is a public research university located in "
    "Lawrence Kansas, with additional campuses in the Kansas City area."
)


def test_bm25_rank_favors_passages_with_more_query_terms():
    passages = [
        "cats and dogs are common pets",
        "the seating capacity of the arena is large",
        "totally unrelated text about weather patterns",
    ]
    ranked = bm25_rank("seating capacity arena", passages)
    assert ranked[0][0] == 1  # the arena/seating passage should rank first
    assert ranked[0][1] > 0


def test_bm25_rank_empty_query_or_passages():
    assert bm25_rank("", ["some text"]) == []
    assert bm25_rank("query", []) == []


def test_slice_context_returns_best_matching_passage():
    result = slice_context(DOC, "seating capacity Colisee", top_k=1, passage_chars=200)
    assert "Colisee" in result or "seating" in result.lower()
    assert "Kansas" not in result


def test_slice_context_no_match_returns_empty():
    result = slice_context(DOC, "xyzzy quux nonexistent term", top_k=2, passage_chars=200)
    assert result == ""


def test_slice_context_preserves_original_document_order():
    result = slice_context(DOC, "Kansas City Lawrence university arena seating", top_k=3, passage_chars=200)
    # if multiple passages are kept, they must appear in the same order as in DOC
    colisee_pos = result.find("Colisee")
    kansas_city_pos = result.find("Kansas City")
    university_pos = result.find("University")
    positions = [p for p in (colisee_pos, kansas_city_pos, university_pos) if p >= 0]
    assert positions == sorted(positions)
