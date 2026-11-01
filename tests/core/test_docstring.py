from agent_adapter.core._docstring import parse_docstring


def test_parse_docstring_splits_description_and_args():
    doc = """Search articles.

    Longer explanation.

    Args:
        query: The search text.
        top_k (int): How many hits
            to return.

    Returns:
        A list of hits.
    """
    description, params = parse_docstring(doc)
    assert description == "Search articles.\n\nLonger explanation."
    assert params == {"query": "The search text.", "top_k": "How many hits to return."}


def test_parse_docstring_none_is_empty():
    assert parse_docstring(None) == ("", {})


def test_parse_docstring_without_sections_is_all_description():
    assert parse_docstring("Just a summary.") == ("Just a summary.", {})


def test_parse_docstring_ignores_raises_section():
    doc = "Do it.\n\nRaises:\n    ValueError: bad input.\n"
    assert parse_docstring(doc) == ("Do it.", {})


def test_parse_docstring_same_indent_non_param_line_continues_current_param():
    doc = "Do it.\n\nArgs:\n    query: The text, e.g.\n    note: still query.\n    top_k: Hits.\n"
    assert parse_docstring(doc, params={"query", "top_k"})[1] == {
        "query": "The text, e.g. note: still query.",
        "top_k": "Hits.",
    }


def test_parse_docstring_without_params_treats_every_entry_as_a_param():
    doc = "Do it.\n\nArgs:\n    query: The text.\n    note: other.\n"
    assert parse_docstring(doc)[1] == {"query": "The text.", "note": "other."}
