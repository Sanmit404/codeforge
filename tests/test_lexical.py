from codeforge.retrieval.lexical import BM25, reciprocal_rank_fusion, tokenize


def test_tokenize_splits_identifier_styles():
    assert tokenize("refresh_token") == ["refresh", "token"]
    assert tokenize("refreshTokenStore") == ["refresh", "token", "store"]


def test_bm25_prefers_the_document_with_the_exact_identifier():
    documents = [
        "def charge_payment(amount): return amount",
        "def refresh_token(session): return session.token",
        "a helper about sessions and users",
    ]

    scores = BM25(documents).scores("refresh_token")

    assert scores.index(max(scores)) == 1
    assert scores[0] == 0.0


def test_fusion_rewards_documents_both_rankings_agree_on():
    dense = [3, 1, 2]
    lexical = [1, 4, 3]

    assert reciprocal_rank_fusion([dense, lexical])[0] == 1


def test_fusion_keeps_documents_only_one_ranking_found():
    assert set(reciprocal_rank_fusion([[1], [2]])) == {1, 2}
