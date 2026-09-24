"""Naming a site out loud, including one nobody hard-coded."""
from __future__ import annotations

import pytest

from jevflow import sites


# -- the curated table still works -------------------------------------------
@pytest.mark.parametrize("said,expect", [
    ("open youtube", "YouTube"),
    ("open you tube", "YouTube"),
    ("go to youtube.com", "YouTube"),
    ("search github for faster whisper", "GitHub"),
    ("open gmail", "Gmail"),
])
def test_known_sites_are_found(said, expect):
    assert sites.shortlist(said)[0].name == expect


def test_x_does_not_match_inside_another_word():
    """The site 'X' once matched inside 'codex' and 'netflix'."""
    names = [s.name for s in sites.shortlist("search for codex on netflix")]
    assert "X" not in names


# -- arbitrary domains --------------------------------------------------------
@pytest.mark.parametrize("said,host", [
    ("open figma.com", "figma.com"),
    ("go to notion.so", "notion.so"),
    ("open figma dot com", "figma.com"),
    ("hey jev go to arxiv dot org", "arxiv.org"),
    ("open news.ycombinator.com", "news.ycombinator.com"),
    ("pull up vercel.app", "vercel.app"),
])
def test_spoken_domains_resolve(said, host):
    site = sites.resolve_domain(said)
    assert site is not None, f"{said!r} resolved to nothing"
    assert site.name == host
    assert site.url == f"https://{host}"


@pytest.mark.parametrize("said", [
    "search for apples",
    "open chrome and look up the weather",
    "close this window",
    "what time is it",
    "open visual studio code",
    # A sentence, not a host. 'me' and 'it' are real TLDs, which is exactly how
    # a naive pattern turns ordinary speech into a web address.
    "tell me about it",
    "remind me dot com is not what i said",
])
def test_ordinary_speech_is_not_a_domain(said):
    assert sites.resolve_domain(said) is None, f"{said!r} was mistaken for a domain"


def test_a_curated_site_wins_over_the_bare_domain():
    """youtube.com should stay YouTube, which knows how to search."""
    site = sites.shortlist("open youtube.com")[0]
    assert site.name == "YouTube"
    assert site.search, "the curated entry lost its search URL"


def test_resolved_domain_is_shortlisted_alongside_known_sites():
    """An address nobody curated still reaches the candidate list."""
    names = [s.name for s in sites.shortlist("open databento.com")]
    assert "databento.com" in names


def test_a_resolved_domain_has_no_search_url():
    """We know the address. We do not know how that site searches."""
    site = sites.resolve_domain("open databento.com")
    assert site.search is None
    assert site.search_url("anything") == "https://databento.com"


def test_resolution_is_case_and_scheme_insensitive():
    for said in ("open HTTPS://Figma.com", "go to www.figma.com", "open Figma.Com"):
        site = sites.resolve_domain(said)
        assert site is not None and site.name.endswith("figma.com"), said


def test_the_table_grew_beyond_the_original_sixteen():
    assert len(sites.SITES) > 16


def test_every_site_search_url_escapes_the_query():
    for s in sites.SITES:
        if s.search:
            assert " " not in s.search_url("two words"), s.name
