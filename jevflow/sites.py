"""Websites you can open or search by name - including ones nobody listed.

"Open YouTube" failed because YouTube is not an installed application - it is a
URL, and the command layer only knew about Start Menu shortcuts. Worse, "on
youtube search up codex" fell through to a plain Google search, which is not
what anyone means.

Sites are ranked in code alongside apps and handed to Jev as one candidate list,
so it makes a single choice: "which of these did they mean?" Code then decides
whether that choice is a program to launch or an address to open.

A curated table can never be complete, so a spoken address is also **resolved in
code** - "open figma.com", "go to arxiv dot org". Resolution is deliberately
narrow, because the failure it must not have is turning an ordinary sentence
into a web address: "tell me about it" contains a real TLD, and "remind me dot
com is not what I said" parses as me.com to anything naive. A written dot needs
no space around it; a *spoken* "dot" only counts right after a word that means
navigate. A curated entry always beats the bare host, because the curated one
knows how that site searches and a bare host does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus, urlsplit


@dataclass(frozen=True)
class Site:
    name: str
    url: str
    search: Optional[str] = None      # {q} is replaced with the query
    aliases: tuple[str, ...] = ()
    resolved: bool = False            # parsed from speech rather than curated

    def search_url(self, query: str) -> str:
        if self.search and query:
            return self.search.replace("{q}", quote_plus(query))
        return self.url


SITES: tuple[Site, ...] = (
    Site("YouTube", "https://www.youtube.com",
         "https://www.youtube.com/results?search_query={q}", ("yt", "youtube com")),
    Site("Google", "https://www.google.com",
         "https://www.google.com/search?q={q}", ("google com",)),
    Site("Gmail", "https://mail.google.com", None, ("mail", "email", "g mail")),
    Site("GitHub", "https://github.com",
         "https://github.com/search?q={q}", ("git hub",)),
    Site("Reddit", "https://www.reddit.com",
         "https://www.reddit.com/search/?q={q}", ()),
    Site("Google Maps", "https://www.google.com/maps",
         "https://www.google.com/maps/search/{q}", ("maps",)),
    Site("Amazon", "https://www.amazon.ca",
         "https://www.amazon.ca/s?k={q}", ()),
    Site("Wikipedia", "https://www.wikipedia.org",
         "https://en.wikipedia.org/w/index.php?search={q}", ("wiki",)),
    Site("X", "https://x.com",
         "https://x.com/search?q={q}", ("twitter",)),
    Site("ChatGPT", "https://chatgpt.com", None, ("chat gpt", "gpt")),
    Site("Claude", "https://claude.ai", None, ()),
    Site("Netflix", "https://www.netflix.com",
         "https://www.netflix.com/search?q={q}", ()),
    Site("Spotify Web", "https://open.spotify.com",
         "https://open.spotify.com/search/{q}", ()),
    Site("Stack Overflow", "https://stackoverflow.com",
         "https://stackoverflow.com/search?q={q}", ("stackoverflow",)),
    Site("Drive", "https://drive.google.com", None, ("google drive",)),
    Site("LinkedIn", "https://www.linkedin.com",
         "https://www.linkedin.com/search/results/all/?keywords={q}", ()),

    # --- added because these are the ones actually asked for -----------------
    Site("Notion", "https://www.notion.so",
         "https://www.notion.so/search?q={q}", ()),
    Site("Figma", "https://www.figma.com", None, ()),
    Site("Vercel", "https://vercel.com", None, ()),
    Site("Instagram", "https://www.instagram.com", None, ("insta", "ig")),
    Site("TikTok", "https://www.tiktok.com",
         "https://www.tiktok.com/search?q={q}", ("tik tok",)),
    Site("WhatsApp Web", "https://web.whatsapp.com", None, ("whatsapp", "whats app")),
    Site("Google Calendar", "https://calendar.google.com", None, ("calendar",)),
    Site("Google Docs", "https://docs.google.com/document", None, ("google docs",)),
    Site("Google Sheets", "https://docs.google.com/spreadsheets", None, ("google sheets",)),
    Site("Outlook", "https://outlook.office.com/mail", None, ()),
    Site("Perplexity", "https://www.perplexity.ai",
         "https://www.perplexity.ai/search?q={q}", ()),
    Site("Hugging Face", "https://huggingface.co",
         "https://huggingface.co/search/full-text?q={q}", ("hugging face", "huggingface")),
    Site("npm", "https://www.npmjs.com",
         "https://www.npmjs.com/search?q={q}", ("npmjs",)),
    Site("PyPI", "https://pypi.org",
         "https://pypi.org/search/?q={q}", ("pypi org",)),
    Site("TradingView", "https://www.tradingview.com",
         "https://www.tradingview.com/symbols/{q}", ("trading view",)),
    Site("arXiv", "https://arxiv.org", None, ()),
    Site("Canva", "https://www.canva.com", None, ()),
    Site("Twitch", "https://www.twitch.tv",
         "https://www.twitch.tv/search?term={q}", ()),
    Site("Hacker News", "https://news.ycombinator.com", None, ("hacker news", "hn")),
    Site("Anthropic Docs", "https://docs.anthropic.com", None, ("anthropic docs",)),
)

_BY_NAME = {s.name.lower(): s for s in SITES}

# Top-level domains a spoken address may end in. Deliberately excludes the ones
# that are really file extensions - "open notes.md" and "run main.sh" are not
# web addresses, and .md and .sh are both real TLDs.
_TLDS = (
    "com", "net", "org", "io", "ai", "co", "dev", "app", "so", "me", "gg", "tv",
    "xyz", "edu", "gov", "ca", "uk", "us", "info", "cloud", "to", "fm", "it",
    "de", "fr", "jp", "au", "in", "eu", "tech", "online", "store", "site", "live",
)
_TLD_RE = "|".join(sorted(_TLDS, key=len, reverse=True))

_LABEL = r"[a-z0-9][a-z0-9-]{0,62}"

# A real dot, with no space around it: "figma.com", "news.ycombinator.com".
_WRITTEN = re.compile(rf"\b(?P<host>(?:{_LABEL}\.)+(?:{_TLD_RE}))\b(?![a-z0-9.-])")

# The word "dot", which only counts directly after a verb that means navigate.
# Without that anchor, "remind me dot com is not what I said" becomes me.com.
_NAV = r"(?:open|go\s+to|goto|visit|navigate\s+to|pull\s+up|bring\s+up|launch|load)"
_SPOKEN = re.compile(
    rf"\b{_NAV}\s+(?P<host>{_LABEL}(?:\s+dot\s+{_LABEL})*\s+dot\s+(?:{_TLD_RE}))\b", re.I)


def _host_of(url: str) -> str:
    """The bare host of a URL, without www, for comparing against a spoken one."""
    h = (urlsplit(url).netloc or "").lower()
    return h[4:] if h.startswith("www.") else h


def _clean_host(host: str) -> str:
    h = re.sub(r"^(?:https?://)?", "", host.strip().lower()).strip(".")
    return h[4:] if h.startswith("www.") else h


def resolve_domain(text: str) -> Optional[Site]:
    """The web address spoken in this sentence, if one really was.

    Returns None far more often than it returns a Site, on purpose. A wrong
    address opens a page nobody asked for; a missed one falls through to the
    curated table and then to a search, which is where it would have gone anyway.
    """
    t = (text or "").lower()
    m = _WRITTEN.search(t)
    if m:
        host = _clean_host(m.group("host"))
    else:
        m = _SPOKEN.search(t)
        if not m:
            return None
        host = _clean_host(re.sub(r"\s+dot\s+", ".", m.group("host")))
    if not host or "." not in host:
        return None
    return Site(name=host, url=f"https://{host}", search=None, resolved=True)


def get(name: str) -> Optional[Site]:
    """A curated site by name.

    Resolved domains are deliberately absent: they are not global state, they
    exist only inside the shortlist that produced them, so a choice can never
    name an address this sentence did not contain.
    """
    return _BY_NAME.get((name or "").strip().lower())


def _norm(text: str) -> str:
    """Speech gives 'youtube.com' and 'you tube'; both should match one site."""
    t = (text or "").lower()
    t = re.sub(r"\b(www\.|https?://)", " ", t)
    t = re.sub(r"\.(com|ca|org|net|ai|io|co\.uk)\b", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def shortlist(text: str, limit: int = 4) -> list[Site]:
    """Sites plausibly named in this sentence, best first.

    A resolved address is appended rather than ranked first: if the speaker said
    youtube.com, the curated YouTube - which knows how YouTube searches - is the
    better answer, and the bare host is then dropped as a duplicate of it.
    """
    t = _norm(text)
    scored: list[tuple[float, Site]] = []
    if t:
        words = set(t.split())
        for s in SITES:
            names = [s.name.lower()] + [a.lower() for a in s.aliases]
            best = 0.0
            for n in names:
                nn = _norm(n)
                if not nn:
                    continue
                if nn == t:
                    best = max(best, 1.0)
                elif re.search(rf"\b{re.escape(nn)}\b", t):
                    best = max(best, 0.9)
                elif len(nn) >= 4 and nn.replace(" ", "") in t.replace(" ", ""):
                    # Length guard: without it the site "X" matched inside
                    # "codex" and "netflix", putting Twitter on every shortlist.
                    best = max(best, 0.75)
                elif set(nn.split()) & words:
                    best = max(best, 0.5)
            if best:
                scored.append((best, s))
    scored.sort(key=lambda bs: (-bs[0], bs[1].name))
    hits = [s for _b, s in scored]

    dom = resolve_domain(text)
    if dom is not None and not any(_host_of(s.url) == dom.name for s in hits):
        hits.append(dom)
    return hits[:limit]
