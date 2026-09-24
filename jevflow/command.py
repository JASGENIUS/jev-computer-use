"""Turn a spoken sentence into an action, safely.

The chain, and who owns each link:

    transcript          speech recognition
    -> candidates       CODE   (wide pass over installed apps and known sites)
    -> which one        JEV    (a typed choice, with confidence)
    -> may it run       CODE   (`policy.py`, never the model)
    -> run it           CODE
    -> did it happen    CODE   (verified against real windows, not assumed)

Jev supplies one judgment: what kind of request this was, and which candidate
was meant. It does not grant permission and it does not execute.

That separation is the whole safety story. It used to rest entirely on absence -
there was no destructive option in the action space to choose - which works
right up until a genuinely useful destructive verb needs to exist. `policy.py`
now adds the other half: a verb can be marked always-confirm, and no certainty
however high lets it run unasked, because the model is never asked that question.

Extraction lives in `parse.py` and thresholds live in `policy.py`. What is left
here is orchestration: assemble candidates, ask once, obey the policy, execute,
and report what actually happened rather than what was attempted.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from jevflow import actions, apps, learned, parse, policy, sites
from jevflow.jev import NONE_OF_THESE, Jev, build_questions, build_state

log = logging.getLogger("jevflow.command")

# Browsers, best first, used when the transcript does not name one.
BROWSERS = ("Google Chrome", "Brave", "Microsoft Edge", "Firefox")

# How close a shortlisted name must be before a `none_of_these` is worth
# turning into a question. Deliberately high: offering the nearest app to
# ANY sound turns a refusal into a guess, and a guess that opens something
# is worse than a clean no.
NEAR_MISS_SCORE = 0.55

# Below this on the "is this an instruction at all" noul, nothing runs whatever
# verb Jev picked. It is deliberately low: its job is to catch a sentence that
# is not a command, not to second-guess a command it merely finds underspecified.
INSTRUCTION_FLOOR = 0.30


@dataclass
class Outcome:
    ok: bool
    action: str
    detail: str
    latency_ms: float = 0.0
    verified: Optional[bool] = None


@dataclass
class _Plan:
    """A decided action, held so that "yes" runs the thing that was offered."""
    verb: str
    label: str
    run: Any            # a zero-argument callable returning an Outcome
    # Force the question regardless of certainty. Used when code can see a
    # near-miss that Jev will not vouch for: neither is wrong, so neither is
    # overruled - the person is asked.
    force_confirm: bool = False


def _browser_apps(prefer: str = "") -> list[apps.App]:
    found: list[apps.App] = []
    for name in ([prefer] if prefer else []) + list(BROWSERS):
        for a in apps.shortlist(name, limit=1):
            if a.name not in [c.name for c in found]:
                found.append(a)
    return found


def site_label(site: sites.Site, app_names: list[str]) -> str:
    """What to call a site in the option list Jev chooses from.

    Normally its own name. But an installed app can share it - "Claude" is both
    a desktop app and claude.ai - and the list then contains the same word
    twice. A choice between two identical labels is not a choice: Jev picks
    "Claude", the code resolves the first match, and the website wins every
    time however clearly the app was asked for.

    So a colliding site is named by its host instead. "claude.ai" and "Claude"
    are two things a person can actually choose between, and it is how someone
    would say it out loud anyway.
    """
    if any(site.name.lower() == (n or "").lower() for n in app_names):
        return sites._host_of(site.url)
    return site.name


def _strip_site_words(query: str, site: sites.Site) -> str:
    """Remove the site's own name from the front of a captured query."""
    import re
    q = query.strip()
    for n in sorted([site.name] + list(site.aliases), key=len, reverse=True):
        q = re.sub(rf"^\s*(on\s+|in\s+)?{re.escape(n)}(\.com|\.ca|\.org)?\s*(for|about)?\s+",
                   "", q, flags=re.I)
    return q.strip(" .,")


def _windows_for(exe: str) -> int:
    try:
        return len(actions.windows_for_exe(exe))
    except Exception:
        return 0


def _window_titled(name: str) -> Optional[int]:
    """A visible window whose title mentions this app, or None.

    The fallback for apps with no exe of their own - Store apps, which on
    Windows 11 includes Notepad. Longest title match wins so "Notepad" does
    not grab "Notepad++".
    """
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    try:
        windows = actions.visible_windows()
    except Exception:
        return None
    exact = [h for h, title in windows if (title or "").strip().lower() == wanted]
    if exact:
        return exact[0]
    holding = [(len(title or ""), h) for h, title in windows
               if wanted in (title or "").lower()]
    return min(holding)[1] if holding else None


def _verify_launched(app: apps.App, before: int, timeout_s: float = 12.0) -> Optional[bool]:
    """Did a window actually appear? Unknown is reported as unknown, not success."""
    if not app.exe:
        return None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if len(actions.windows_for_exe(app.exe)) > before:
                return True
        except Exception:
            return None
        time.sleep(0.4)
    return False



# -- which steps may share the machine ----------------------------------------
# Opening three apps is three independent jobs. Running them one at a time
# means waiting for each in turn, and an app that never shows a window costs a
# 12-second timeout - so a sequence of slow launches adds up fast.
#
# But only SOME steps are independent. Searching, typing and switching all take
# the keyboard focus and then send keystrokes to whatever currently has it. Two
# of those at once means keystrokes landing in the wrong window, which is the
# precise failure this whole layer exists to prevent. Closing a window changes
# what is focused, so it counts too.
#
# The list is an ALLOW-list. A verb nobody has classified is treated as
# focus-taking, because the cost of being wrong in that direction is a slower
# sequence, and the cost of being wrong the other way is text typed into the
# wrong document.
FOCUS_FREE_VERBS = frozenset({"open_app", "media_control"})


def is_independent(verb: str) -> bool:
    """May this step run alongside others? Unknown verbs: no."""
    return (verb or "") in FOCUS_FREE_VERBS


def _safely(plan) -> "Outcome":
    """Run one step, turning an exception into a reported failure.

    A step that raises inside a thread pool would otherwise surface as a
    traceback from `pool.map` and take the whole sequence down without saying
    which step did it.
    """
    try:
        return plan.run()
    except Exception as exc:
        log.exception("[cmd] step %r raised", getattr(plan, "label", "?"))
        return Outcome(False, "step_raised",
                       f"{getattr(plan, 'label', 'a step')} failed: {exc}")


def group_plans(plans: list, parallel: bool = False) -> list[list]:
    """Consecutive independent steps become one group; everything else is alone.

    Order across groups is preserved exactly, because "open Chrome then search
    in it" only works in that order.

    **`parallel` is off by default**, so every step is its own group and they
    run strictly one after another. Batching the opens was faster on paper and
    wrong in the room: "open Chrome and then search for X and then open
    Modrinth" launched Chrome and Modrinth together, so three windows appeared
    in a scramble and the search raced the browser it was meant to run in.
    Watching it happen, the order you SAY is the order you expect to see, and a
    couple of saved seconds do not buy anything you wanted.
    """
    if not parallel:
        return [[plan] for plan in plans]
    groups: list[list] = []
    for plan in plans:
        if is_independent(getattr(plan, "verb", "")) and groups \
                and all(is_independent(getattr(p, "verb", "")) for p in groups[-1]):
            groups[-1].append(plan)
        else:
            groups.append([plan])
    return groups

@dataclass
class Commander:
    jev: Optional[Jev] = None
    dry_run: bool = False
    pending: Optional[policy.Pending] = None
    # What was last opened, and the words that asked for it. A correction needs
    # something to correct: "no, I meant LocalSend" is meaningless without
    # knowing that "local send" was the phrase that went wrong.
    last_open: Optional[tuple] = None
    # Run independent steps at the same time. Off: in use, three windows
    # appeared in a scramble and a search raced the browser it belonged in,
    # and the order you say is the order you expect to see.
    parallel: bool = False
    # Never stop to ask yes/no. When the policy would have asked, it acts
    # instead - or, for a verb marked always_confirm, refuses. See _prepare.
    never_ask: bool = True
    # Everything opened since this session started, oldest first. "Close the
    # three apps you just opened" names nothing, so the shortlist comes back
    # empty and the step used to die with "nothing installed matches" - which
    # reads as a broken tool rather than as a sentence it does not handle.
    # Only apps that actually opened are recorded, so a failed launch is not
    # something you can be offered back.
    opened: list = field(default_factory=list)
    # Called as each step of a sequence starts, so the capsule can say "2 of 4"
    # instead of sitting frozen while three apps open behind it.
    on_progress: Optional[Any] = None
    _plans: dict[str, _Plan] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.jev is None:
            self.jev = Jev()

    # -- entry point -------------------------------------------------------
    def handle(self, transcript: str, focused_window: str = "",
               focused_hwnd: Optional[int] = None) -> Outcome:
        answered = self._answer_pending(transcript)
        if answered is not None:
            return answered

        corrected = self._answer_correction(transcript)
        if corrected is not None:
            return corrected

        # Strip the polite run-up FIRST. Every verb pattern is anchored at the
        # start of the sentence, so "now let's close Notepad" matched nothing
        # and the step died saying nothing was named. Doing it here means every
        # verb gets it, and the splitter below sees a sentence that starts with
        # a command.
        transcript = parse.strip_framing(transcript)

        # One sentence can ask for several things. Every step is planned before
        # ANY of them runs, so a sequence that cannot complete never
        # half-completes - three of four done, with nothing said about the
        # fourth, is worse than refusing the lot.
        steps = parse.split_steps(transcript)
        if len(steps) > 1:
            return self._handle_sequence(steps, focused_window, focused_hwnd)
        return self._handle_one(transcript, focused_window, focused_hwnd)

    def _handle_one(self, transcript: str, focused_window: str = "",
                    focused_hwnd: Optional[int] = None) -> Outcome:
        prepared = self._prepare(transcript, focused_window, focused_hwnd)
        if isinstance(prepared, Outcome):
            return prepared
        plan, band, ms, certain = prepared
        if band == policy.REFUSE:
            return Outcome(False, "unsure", f"not sure enough to {plan.label} - say it again ({certain:.2f})", ms)
        if band == policy.CONFIRM:
            self.pending = policy.Pending(verb=plan.verb, prompt=plan.label,
                                          payload={"id": id(plan)})
            self._plans = {str(id(plan)): plan}
            return Outcome(False, "needs_confirm", f"{plan.label}? say yes or no", ms)
        return plan.run()

    def _prepare(self, transcript: str, focused_window: str = "",
                 focused_hwnd: Optional[int] = None):
        """Decide what one instruction means, WITHOUT doing it.

        Returns either a ready `_Plan` with the band that governs it, or an
        `Outcome` explaining why there is nothing to do. Separating this from
        execution is what lets a sequence be fully planned up front.
        """
        # "write a poem about apples" needs words that do not exist yet, and
        # nothing here can invent them - Jev answers typed questions and cannot
        # return prose. Checked BEFORE asking, because whether this tool can
        # compose has nothing to do with how the sentence gets classified:
        # asked first, Jev called it open_app and it opened Notepad instead.
        composing = parse.compose_request(transcript)
        if composing:
            return Outcome(False, "cannot_compose",
                           f"I can't write {composing} - I can only type words "
                           f"you say out loud")

        site_hits = sites.shortlist(transcript)
        query = parse.app_query(transcript)
        query_text = parse.search_query(transcript)
        typed = parse.type_request(transcript)
        media = parse.media_action(transcript)
        browser_hint = next((b for b in BROWSERS
                             if b.split()[-1].lower() in transcript.lower()), "")

        cand_apps = self._candidate_apps(transcript, query, query_text, typed,
                                         site_hits, browser_hint)
        app_names = [a.name for a in cand_apps]
        site_labels = [site_label(s, app_names) for s in site_hits]
        names = site_labels + app_names
        # Jev cannot see the taskbar. Until it was told, "switch to Obsidian"
        # always came back as open_app - a model has no way to tell something
        # running from something merely installed.
        open_now = apps.running(cand_apps)
        log.info("[cmd] %r -> app_q %r search_q %r -> %s (open: %s)",
                 transcript, query, query_text, names, open_now or "none")

        try:
            res = self.jev.ask(
                build_state(transcript, names, focused_window, query_text, open_now),
                build_questions(names))
        except Exception as exc:
            return Outcome(False, "error", f"Jev unavailable: {exc}")

        ans = res["answers"]
        intent = ans.get("intent") or {}
        verb = intent.get("choice") or "unclear"
        # Two different numbers, and only one of them is about the decision.
        #
        # `certain` is a noul: "the transcript is a clear, specific instruction
        # that can be acted on now". `intent.confidence` is how sure Jev is of
        # the VERB. The policy bands decide whether to run a verb, so the verb
        # confidence is the one they should read.
        #
        # Gating on the noul alone broke typing while STREAMING, measured
        # 2026-09-22. "type in hello" fires while you are still speaking, so Notepad
        # has not finished opening; the noul collapsed to 0.53-0.60 purely
        # because there was no target yet, while the intent stayed at 0.83-0.91
        # throughout. The result was a yes/no prompt on every dictated line -
        # the gate punishing the streaming feature for working.
        #
        # The noul is kept as a FLOOR, because it is genuinely good at the
        # question it answers: a rambling sentence that is not an instruction
        # at all scores near zero on it, and nothing should run on that.
        certain = float((ans.get("certain") or {}).get("noul") or 0.0)
        confidence = float(intent.get("confidence") or 0.0)
        if not confidence:
            # A response with no per-choice confidence: fall back to the noul
            # rather than reading a missing field as "completely unsure" and
            # refusing everything.
            confidence = certain
        certainty = confidence if certain >= INSTRUCTION_FLOOR else certain
        chosen = (ans.get("app") or {}).get("choice")
        ms = res["latency_ms"]
        log.info("[cmd] intent=%s app=%s intent_conf=%.2f certain=%.2f -> gate %.2f",
                 verb, chosen, confidence, certain, certainty)

        if verb == "unclear":
            return Outcome(False, verb, "not understood", ms)
        # `type_text` is Jev's verb for "type where the cursor already is", and
        # it used to be refused with "not an app request" - so JCU could never
        # type anything at all. After "open Notepad", "type the word apples"
        # means precisely this: the target is whatever has focus, which is the
        # thing that was just opened.
        if verb == "type_text":
            verb = "type_into"

        # --- policy. Code decides, not the model. --------------------------
        p = policy.for_verb(verb)
        if p is None:
            # An intent with no policy does nothing. It must never fall through
            # to a permissive default - a verb someone forgot to wire up would
            # then run as the most capable thing available.
            log.warning("[cmd] no policy for intent %r; refusing", verb)
            return Outcome(False, "no_policy", f"{verb} is not something I do", ms)

        plan = self._plan(verb, chosen, cand_apps, site_hits, query, query_text,
                          typed, media, browser_hint, focused_hwnd, ms, transcript,
                          site_labels=site_labels)
        if isinstance(plan, Outcome):
            return plan
        band = policy.decide_with(p, certainty)
        if getattr(plan, "force_confirm", False):
            band = policy.CONFIRM
        if band == policy.CONFIRM and self.never_ask:
            # JCU does not ask yes/no. Ever. That is a requirement from real
            # use, not a default. Removing prompts one at a time kept missing the next one,
            # so this is the choke point instead: the ONLY two places that
            # ever set `self.pending` both require band == CONFIRM, and
            # CONFIRM cannot get past this line.
            #
            # A verb marked always_confirm is the one exception in the other
            # direction - a forced kill has no undo, so it is REFUSED rather
            # than run unasked. It still never asks; it just does not happen.
            band = policy.REFUSE if p.always_confirm else policy.ACT
            log.info("[cmd] never_ask: %s -> %s", plan.label, band)
        return plan, band, ms, certainty

    # -- several instructions in one sentence ------------------------------
    def _handle_sequence(self, steps: list[str], focused_window: str,
                         focused_hwnd: Optional[int]) -> Outcome:
        """Plan every step, then run them in order, stopping at the first failure."""
        plans: list[_Plan] = []
        total_ms = 0.0
        needs_confirm = False
        for i, step in enumerate(steps, 1):
            prepared = self._prepare(step, focused_window, focused_hwnd)
            if isinstance(prepared, Outcome):
                # One unusable step kills the whole sequence, BEFORE anything
                # has run. Executing the rest would do a different job from the
                # one that was asked for, silently.
                # The message is what you actually read, on a capsule, in
                # passing. "Step Unclear" reads as the program not
                # understanding English; naming the step and the reason reads
                # as a thing that went wrong in a specific place.
                return Outcome(False, "couldnt_do_step",
                               f"couldn't do \"{step}\" ({prepared.detail}) "
                               f"- step {i} of {len(steps)}, nothing was done",
                               total_ms)
            plan, band, ms, certain = prepared
            total_ms += ms
            if band == policy.REFUSE:
                return Outcome(False, "unsure",
                               f"step {i} of {len(steps)}: not sure enough to {plan.label} "
                               f"(certainty {certain:.2f}) - nothing was done",
                               total_ms)
            needs_confirm = needs_confirm or band == policy.CONFIRM
            plans.append(plan)

        if not plans:
            return Outcome(False, "unclear", "nothing to do", total_ms)

        # ONE question for the whole sequence. Asking per step turns a
        # four-part instruction into four interrogations.
        if needs_confirm:
            seq = _Plan("sequence", self._describe(plans),
                        lambda: self._run_sequence(plans, total_ms))
            self.pending = policy.Pending(verb="sequence", prompt=seq.label,
                                          payload={"id": id(seq)})
            self._plans = {str(id(seq)): seq}
            return Outcome(False, "needs_confirm",
                           f"{seq.label}? say yes or no", total_ms)
        return self._run_sequence(plans, total_ms)

    @staticmethod
    def _describe(plans: list[_Plan]) -> str:
        return f"{len(plans)} things: " + ", then ".join(p.label for p in plans)

    def _run_sequence(self, plans: list[_Plan], ms: float) -> Outcome:
        """In order, stopping at the first failure - and always saying so.

        Half a sequence with no word about the rest is the worst outcome
        available here: the person walks away believing it all happened.
        """
        from concurrent.futures import ThreadPoolExecutor

        done: list[str] = []
        step = 0
        for group in group_plans(plans, parallel=self.parallel):
            if self.on_progress is not None:
                try:
                    # A parallel group is one event, not several: three apps
                    # opening at once is not "step 1 of 3" three times.
                    label = (group[0].label if len(group) == 1
                             else " + ".join(p.label for p in group))
                    self.on_progress(step + 1, len(plans), label)
                except Exception as exc:
                    # A broken progress callback must never stop the work.
                    log.debug("[cmd] progress callback failed: %s", exc)

            if len(group) == 1:
                results = [_safely(group[0])]
            else:
                # Independent steps overlap. Reported in the order asked for,
                # not the order they happened to finish - "opened Spotify,
                # Notepad, Chrome" for "notepad, chrome, spotify" reads like it
                # did something else.
                with ThreadPoolExecutor(max_workers=min(4, len(group))) as pool:
                    results = list(pool.map(_safely, group))

            for plan, out in zip(group, results):
                step += 1
                log.info("[cmd] step %d/%d %s -> ok=%s %s",
                         step, len(plans), plan.label, out.ok, out.detail)
                if not out.ok:
                    did = ("did: " + "; ".join(done) + ". ") if done else "nothing ran. "
                    return Outcome(False, "sequence_failed",
                                   f"{did}stopped at step {step} of {len(plans)} "
                                   f"({plan.label}): {out.detail}",
                                   ms, verified=False)
                done.append(out.detail)
        return Outcome(True, "sequence", f"{len(plans)} steps: " + "; ".join(done),
                       ms, verified=True)

    # -- "no, I meant X" ---------------------------------------------------
    def _answer_correction(self, transcript: str) -> Optional[Outcome]:
        """Take the correction, learn it, and open what was actually wanted.

        Deliberately does NOT ask Jev. The person just said which app they
        meant, in as many words - running that back through a classifier only
        creates a second chance to get it wrong.
        """
        wanted = parse.correction_request(transcript)
        if wanted is None:
            return None
        if not self.last_open:
            # Nothing to correct. Learning here would attach the lesson to
            # whatever happened to be said before, which is how a store fills
            # up with nonsense.
            return Outcome(False, "nothing_to_correct",
                           "nothing was opened just now to correct")

        said, _was = self.last_open
        hits = apps.shortlist(wanted, limit=4)
        if not hits:
            return Outcome(False, "no_match", f"nothing installed matches {wanted!r}")
        app = hits[0]
        learned.remember(said, app.name)
        log.info("[cmd] corrected %r -> %r (learned)", said, app.name)
        self.last_open = (said, app.name)
        out = self._do_open(app, 0.0)
        if out.ok:
            return Outcome(out.ok, out.action,
                           f"{out.detail}  (learned: {said!r} means {app.name})",
                           out.latency_ms, out.verified)
        return out

    # -- a question that is already on screen ------------------------------
    def _answer_pending(self, transcript: str) -> Optional[Outcome]:
        """Read this utterance as the answer to the last question, if there was one.

        Deliberately does NOT ask Jev again. The decision was already made and
        shown; re-deriving it would let a second, different answer run in place
        of the one the person actually said yes to.
        """
        pend = self.pending
        if pend is None:
            return None
        verdict = pend.resolve(transcript)
        if verdict == policy.CONFIRMED:
            plan = self._plans.get(str(pend.payload.get("id")))
            self.pending, self._plans = None, {}
            if plan is None:
                return Outcome(False, "expired", "that offer is gone; say it again")
            log.info("[cmd] confirmed by voice: %s", plan.label)
            return plan.run()
        if verdict == policy.DENIED:
            self.pending, self._plans = None, {}
            return Outcome(False, "cancelled", "alright, nothing done")
        # EXPIRED or SUPERSEDED: drop it and treat this as a fresh utterance.
        self.pending, self._plans = None, {}
        if verdict == policy.EXPIRED and (parse.is_affirmation(transcript)
                                          or parse.is_denial(transcript)):
            return Outcome(False, "expired", "too late - say the whole thing again")
        return None

    # -- the wide pass, in code -------------------------------------------
    def _candidate_apps(self, transcript, query, query_text, typed, site_hits,
                        browser_hint) -> list[apps.App]:
        if typed is not None:
            # Only when a target was NAMED. Otherwise the words being dictated
            # get hunted for as an application, and "type the word apples"
            # comes back holding Microsoft Word.
            return apps.shortlist(typed.target, limit=4) if typed.target else []
        if query_text and not site_hits:
            return _browser_apps(browser_hint)
        closing = parse.close_target(transcript)
        if closing:
            return apps.shortlist(closing, limit=4)
        if query:
            return apps.shortlist(query, limit=4)
        return []

    # -- turn a choice into something runnable -----------------------------
    def _plan(self, verb, chosen, cand_apps, site_hits, query, query_text, typed,
              media, browser_hint, focused_hwnd, ms, transcript="",
              site_labels=None):
        """Either a _Plan, or an Outcome explaining why there is nothing to do."""
        app = next((a for a in cand_apps if a.name == chosen), None)
        # Looked up in THIS sentence's shortlist, never in a global table: a
        # choice can then only ever name something code actually offered.
        labels = site_labels if site_labels is not None else [s.name for s in site_hits]
        site = next((s for s, label in zip(site_hits, labels) if label == chosen), None)

        if verb == "media_control":
            if not media:
                return Outcome(False, "no_media_key", "not sure which control that is", ms)
            return _Plan(verb, media.replace("_", " "),
                         lambda: self._do_media(media, ms))

        if verb == "close_window":
            wanted = parse.closes_recent(transcript)
            if wanted is not None:
                # Points backwards rather than naming anything: "close the
                # three apps you just opened". 0 means all of them.
                targets = self.opened[-wanted:] if wanted else list(self.opened)
                if not targets:
                    return Outcome(False, "nothing_opened",
                                   "I have not opened anything yet", ms)
                listed = ", ".join(a.name for a in targets)
                # This used to confirm every time, on the grounds that one
                # wrong word could close something you were working in. The prompt
                # went because the risk is small: it only ever closes
                # apps JCU ITSELF opened in this session, never anything you
                # opened by hand, and it is WM_CLOSE - an app with unsaved
                # work still gets to put its own dialog up. The safety was
                # already there; the question on top was just noise.
                return _Plan(verb, f"close {listed}",
                             lambda t=tuple(targets): self._close_many(t, ms))
            return _Plan(verb, f"close {app.name if app else 'this window'}",
                         lambda: self._do_close(app, focused_hwnd, ms))

        if verb == "switch_to":
            if app is None:
                return Outcome(False, "no_match", f"nothing installed matches {query!r}", ms)
            return _Plan(verb, f"switch to {app.name}", lambda: self._do_switch(app, ms))

        if verb == "type_into":
            if typed is None or not typed.text:
                return Outcome(False, "no_text", "could not tell what to type", ms)
            return _Plan(verb, f"type {typed.text!r} into {app.name if app else 'this window'}",
                         lambda: self._do_type(typed, app, ms))

        if site is not None and verb in ("open_app", "web_search"):
            q = _strip_site_words(query_text, site) if query_text else ""
            searching = verb == "web_search" and q and site.search
            url = site.search_url(q) if searching else site.url
            label = f"{q} on {site.name}" if searching else site.name
            return _Plan(verb, label,
                         lambda: self._do_site(site, url, label, browser_hint, ms))

        if verb == "web_search":
            if not query_text:
                return Outcome(False, "no_query", "could not tell what to search for", ms)
            return _Plan(verb, f"search for {query_text}",
                         lambda: self._do_search(query_text, app, cand_apps, ms))

        # open_app
        #
        # A FAILED match still records the phrase. That is the moment teaching
        # matters most - "open the sender" finds nothing, and "no, I meant
        # LocalSend" is the obvious next sentence. Recording only successful
        # opens left the correction with nothing to attach to in exactly the
        # case it was built for.
        if query:
            self.last_open = (query.strip(), None)
        if not cand_apps and not site_hits:
            return Outcome(False, "no_match", f"nothing installed matches {query!r}", ms)
        if chosen in (None, NONE_OF_THESE) and verb == "web_search" and query_text:
            # "search for pizza near me" names no application, so the honest
            # answer to "which of these browsers did you mean" IS none of them -
            # and Jev says so at 1.00 confidence. The code then read that as
            # "nothing matches" and did nothing at all, which is how "open
            # Chrome and search for <a local business>" opened Chrome and then sat
            # there. Measured in a real session log, 2026-09-22 22:51.
            #
            # A web search does not need a named app. Not naming one means "use
            # my browser", so use it.
            browser = next(iter(_browser_apps(browser_hint)), None)
            if browser is not None:
                log.info("[cmd] web_search with no app named; using %s", browser.name)
                return _Plan(verb, f"search for {query_text}",
                             lambda b=browser, q=query_text:
                                 self._do_search(q, b, cand_apps, ms))
        if chosen in (None, NONE_OF_THESE):
            # Jev will not vouch for any of them - usually right, but the
            # recogniser mangles names ("Minecraft launcher" came through as
            # "Minecraft WATCHER") and code can still see the near-miss. Rather
            # than overrule either side, ask.
            near = max(((apps._score(query, a), a) for a in cand_apps),
                       key=lambda sa: sa[0], default=(0.0, None))
            if near[1] is not None and near[0] >= NEAR_MISS_SCORE:
                log.info("[cmd] %r -> none_of_these, but %r scores %.2f; asking",
                         query, near[1].name, near[0])
                return _Plan(verb, f"did you mean {near[1].name}",
                             lambda a=near[1], q=query: self._open_and_learn(a, q, ms),
                             force_confirm=True)
            return Outcome(False, "no_match",
                           f"nothing installed matches {query!r}", ms)
        if app is None:
            # Jev named something code never offered. That should be impossible,
            # so it is a bug rather than a request - and nothing gets launched.
            log.warning("[cmd] %r was chosen but never shortlisted", chosen)
            return Outcome(False, "no_match", f"{chosen!r} is not in the shortlist", ms)
        return _Plan(verb, f"open {app.name}",
                     lambda: self._open_and_remember(app, query, ms))

    # -- execution ---------------------------------------------------------
    def _open_and_learn(self, app: apps.App, said: str, ms: float) -> Outcome:
        """Open the near-miss they just confirmed, and remember it.

        They have now told us in as many words that this sound means this app.
        Not recording that would ask the same question every single time.
        """
        learned.remember(said, app.name)
        log.info("[cmd] confirmed near-miss: %r means %r", said, app.name)
        return self._open_and_remember(app, said, ms)

    def _open_and_remember(self, app: apps.App, said: str, ms: float) -> Outcome:
        """Open it, and keep the phrase that asked for it.

        Without the phrase there is nothing for "no, I meant X" to attach the
        lesson to, and the correction would have to guess what it was fixing.
        """
        self.last_open = ((said or "").strip(), app.name)
        out = self._do_open(app, ms)
        if out.ok:
            # Only on success. Being offered back something that never opened
            # would make "close the ones you opened" close the wrong window.
            self.opened = [a for a in self.opened if a.name != app.name] + [app]
        return out

    def _do_open(self, app: apps.App, ms: float) -> Outcome:
        if self.dry_run:
            return Outcome(True, "would_open", app.name, ms)
        # A Store app has no exe, so its windows cannot be counted. Snapshot
        # every window instead and look for a new one wearing its name.
        before = _windows_for(app.exe) if app.exe else 0
        before_hwnds = set() if app.exe else {h for h, _t in actions.visible_windows()}
        ok, detail = apps.launch(app)
        if not ok:
            return Outcome(False, "launch_failed", detail, ms)
        verified = (_verify_launched(app, before) if app.exe
                    else actions.wait_for_new_titled(app.name, before_hwnds))
        log.info("[cmd] launched %s -> window appeared: %s", app.name, verified)
        if verified is False:
            return Outcome(False, "no_window",
                           f"{app.name} was started but no window appeared", ms,
                           verified=False)
        return Outcome(True, "opened", app.name, ms, verified=verified)

    def _do_site(self, site: sites.Site, url: str, label: str, browser_hint: str,
                 ms: float) -> Outcome:
        if self.dry_run:
            return Outcome(True, "would_open_site", f"{label}  [{url}]", ms)
        opened = actions.open_url(url, _browser_apps(), apps.launch, prefer=browser_hint)
        if not opened.ok:
            return Outcome(False, "no_browser", opened.note or "no browser opened", ms)
        # A browser window existing is not proof the address loaded. Confirm it
        # by title if we can, and say "unknown" when we cannot - reporting
        # success because a process was spawned is exactly the lie to avoid.
        needle = (site.name.split()[0] if not site.resolved
                  else site.name.split(".")[0])
        title = actions.wait_for_title(opened.exe, needle, timeout_s=8.0)
        detail = f"{label} in {opened.browser}"
        if opened.note:
            detail += f"  ({opened.note})"
        return Outcome(True, "opened_site", detail, ms,
                       verified=True if title else None)

    def _do_media(self, action: str, ms: float) -> Outcome:
        if self.dry_run:
            return Outcome(True, "would_press", action.replace("_", " "), ms)
        if not actions.press_media(action):
            return Outcome(False, "no_media_key", f"no key for {action}", ms)
        # Nothing on this machine can confirm that a song changed, so this
        # stays unknown forever rather than being quietly called success.
        return Outcome(True, "pressed", action.replace("_", " "), ms, verified=None)

    def _do_switch(self, app: apps.App, ms: float) -> Outcome:
        wins = actions.windows_for_exe(app.exe)
        if not wins:
            return Outcome(False, "not_open", f"{app.name} is not open", ms)
        if self.dry_run:
            return Outcome(True, "would_switch", app.name, ms)
        ok = actions.focus_window(wins[0][0])
        return Outcome(ok, "switched" if ok else "focus_failed",
                       app.name if ok else f"could not bring {app.name} to the front",
                       ms, verified=ok)

    def _do_type(self, typed: parse.TypeRequest, app: Optional[apps.App],
                 ms: float) -> Outcome:
        if self.dry_run:
            return Outcome(True, "would_type",
                           f"{typed.text!r} into {app.name if app else 'this window'}", ms)
        hwnd = None
        if app is not None:
            wins = actions.windows_for_exe(app.exe)
            if not wins:
                return Outcome(False, "not_open", f"{app.name} is not open", ms)
            hwnd = wins[0][0]
            if not actions.focus_window(hwnd):
                return Outcome(False, "focus_failed",
                               f"could not bring {app.name} to the front", ms)
            time.sleep(0.25)
        else:
            hwnd, _t = actions.foreground()
        ok, why = actions.type_text(typed.text, expect_hwnd=hwnd)
        if not ok:
            return Outcome(False, "type_failed", why, ms)
        # The text was sent to the window that was verified as focused. Whether
        # the application put it anywhere useful is not observable from here.
        return Outcome(True, "typed", f"{typed.text!r} into {why}", ms, verified=None)

    def _close_many(self, targets, ms: float) -> Outcome:
        """Close several named apps, and say honestly which ones did not go.

        Newest first, because closing the thing opened last is the one most
        likely to still be in front. Every app is attempted even if an earlier
        one refuses - a half-finished close that stops silently is worse than
        one that finishes and reports what it could not do.
        """
        closed, refused = [], []
        for app in reversed(list(targets)):
            out = self._do_close(app, None, ms)
            (closed if out.ok else refused).append(app.name)
        if closed:
            self.opened = [a for a in self.opened if a.name not in closed]
        if not closed:
            return Outcome(False, "none_closed",
                           f"could not close {', '.join(refused)}", ms)
        detail = ", ".join(closed)
        if refused:
            # Named, not swallowed. An app holding an unsaved-work dialog is
            # the usual reason, and you need to know which one is waiting.
            return Outcome(True, "closed_some",
                           f"closed {detail}; {', '.join(refused)} would not close", ms)
        return Outcome(True, "closed", detail, ms)

    def _do_close(self, app: Optional[apps.App], focused_hwnd: Optional[int],
                  ms: float) -> Outcome:
        """Close a named application's window, or the one you were just in.

        Only ever WM_CLOSE, so an app with unsaved work still gets to ask. If it
        asks, this reports that it did NOT close rather than claiming success.
        """
        if app is not None and app.exe:
            wins = actions.windows_for_exe(app.exe)
            if not wins:
                return Outcome(False, "not_open", f"{app.name} is not open", ms)
            if self.dry_run:
                return Outcome(True, "would_close", app.name, ms)
            ok, detail = actions.close_window(wins[0][0])
            return Outcome(ok, "closed" if ok else "not_closed",
                           app.name if ok else detail, ms, verified=ok)
        if app is not None:
            # A Store app has no exe to match windows against - and Notepad on
            # Windows 11 is one. Without this, "close Notepad" resolved
            # perfectly (Jev at 0.96) and then fell straight through to the
            # focused-window branch, reporting "nothing named, and no window
            # was focused" about a sentence that named something very clearly.
            #
            # Matched on the window title instead, which is how launching
            # already verifies these same windows appeared.
            hwnd = _window_titled(app.name)
            if hwnd is None:
                return Outcome(False, "not_open", f"{app.name} is not open", ms)
            if self.dry_run:
                return Outcome(True, "would_close", app.name, ms)
            ok, detail = actions.close_window(hwnd)
            return Outcome(ok, "closed" if ok else "not_closed",
                           app.name if ok else detail, ms, verified=ok)
        if not focused_hwnd:
            return Outcome(False, "no_target", "nothing named, and no window was focused", ms)
        if self.dry_run:
            return Outcome(True, "would_close", "the focused window", ms)
        ok, detail = actions.close_window(focused_hwnd)
        return Outcome(ok, "closed" if ok else "not_closed", detail, ms, verified=ok)

    # -- a browser tab and a search ----------------------------------------
    def _do_search(self, query: str, app: Optional[apps.App],
                   cand_apps: list[apps.App], ms: float) -> Outcome:
        """Focus or launch a browser, open a tab, type the query, press Enter.

        Each step is verified before the next one runs, because a keystroke sent
        to a window that is not really focused lands in whatever IS focused -
        which is how automation types into the wrong application.
        """
        if self.dry_run:
            return Outcome(True, "would_search",
                           f"{app.name if app else 'browser'}: {query}", ms)
        if app is None:
            app = next(iter(_browser_apps()), None)
        if app is None:
            return Outcome(False, "no_browser", "no browser found on this machine", ms)

        # Try the chosen browser, then the others. A browser can exit silently -
        # Chrome does exactly that on this machine - and a tool that reports
        # success because a process was spawned is lying.
        known_dead = actions.dead_browsers()
        order = [app] + [b for b in _browser_apps() if b.name != app.name]
        order.sort(key=lambda b: b.name in known_dead)
        hwnd, tried, downgraded = None, [], []
        for cand in order:
            existing = actions.windows_for_exe(cand.exe)
            if existing:
                app, hwnd = cand, existing[0][0]
                break
            ok, _detail = apps.launch(cand)
            tried.append(cand.name)
            if not ok:
                continue
            got = actions.wait_for_window(cand.exe, timeout_s=12.0)
            if got is not None:
                app, hwnd = cand, got[0]
                time.sleep(1.2)      # let it paint before typing at it
                break
            actions.mark_browser_dead(cand.name)
            downgraded.append(cand.name)
            log.warning("[cmd] %s launched but showed no window; trying the next browser",
                        cand.name)
        if hwnd is None:
            return Outcome(False, "no_window",
                           "no browser opened (tried " + ", ".join(tried) + ")", ms)

        if not actions.focus_window(hwnd):
            return Outcome(False, "focus_failed", f"could not bring {app.name} to the front", ms)
        if not actions.press(actions.VK_CONTROL, actions.VK_T, expect_hwnd=hwnd):
            return Outcome(False, "focus_lost", "focus moved before the new tab", ms)
        time.sleep(0.35)
        ok, why = actions.type_text(query, expect_hwnd=hwnd)
        if not ok:
            return Outcome(False, "type_failed", why, ms)
        time.sleep(0.15)
        actions.press(actions.VK_RETURN, expect_hwnd=hwnd)

        needle = query.split()[0] if query.split() else ""
        verified = actions.wait_for_title(app.exe, needle, timeout_s=8.0) is not None
        detail = f"{query} in {app.name}"
        if downgraded:
            detail += f"  ({', '.join(downgraded)} did not start)"
        return Outcome(True, "searched", detail, ms, verified=verified)
