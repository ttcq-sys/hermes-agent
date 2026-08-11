from gateway.kanban_watchers import _wiki_disposition_suffix


def test_no_candidate_has_no_visible_suffix():
    assert _wiki_disposition_suffix({"status": "none", "reason": "unchanged"}) == ""


def test_routed_candidate_is_visible_in_terminal_notification():
    assert _wiki_disposition_suffix(
        {"status": "candidate-routed", "reason": "reusable rule"}
    ) == "\nWiki: 중요사항 후보를 지식참모 검토로 넘겼습니다."


def test_saved_candidate_is_visible_in_terminal_notification():
    assert _wiki_disposition_suffix(
        {"status": "saved", "reason": "merged and read back"}
    ) == "\nWiki: 승인된 중요사항이 정본에 반영됐습니다."
