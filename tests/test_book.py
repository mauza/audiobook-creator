import json
import subprocess

import numpy as np
import pytest
from ebooklib import epub

from audiobook_creator.file_handlers import (
    _clean_language,
    _normalize_for_tts,
    extract_epub_metadata,
    extract_text_from_epub,
)
from audiobook_creator.tts import TTSConverter, _spoken_title


def _page(name, title, body):
    page = epub.EpubHtml(title=title, file_name=name)
    page.content = f"<html><body>{body}</body></html>"
    return page


@pytest.fixture
def book_path(tmp_path):
    book = epub.EpubBook()
    book.set_identifier("id1")
    book.set_title("Calm Mind")
    book.add_author("Ann Writer")
    book.add_metadata("DC", "date", "2020")
    cover = _page("01-cover.xhtml", "Cover", "<p>cover art</p>")
    copyright_page = _page("02-copyright.xhtml", "Copyright", "<p>All rights reserved.</p>")
    part = _page(
        "03-part01.xhtml", "Part I",
        '<h1 class="part_number">Part I</h1><h1 class="part_title">Your Mind</h1>'
        '<p class="pull_quote3">“How we spend our days is how we spend our lives.”</p>'
        '<p class="part_toc">1</p><p class="chap_toc">Thoughts Pass</p>',
    )
    chapter = _page(
        "04-chapter01.xhtml", "1: Thoughts Pass",
        '<h1 class="chapter_number">1</h1><h1 class="chapter_title">Thoughts Pass</h1>'
        "<p>It’s fine to have a thought.</p>"
        # html.parser nests the paragraph after an unclosed <li> inside it.
        "<ol><li><strong>Past mistakes: </strong>We all screw up."
        "<p>Maybe you carry regret.</p></ol>"
        '<p class="listnum"><span class="number">1</span>  Thoughts are not facts.</p>',
    )
    notes = _page("05-notes.xhtml", "Notes", "<h1>Notes</h1><p>1: A citation, 2012.</p>")
    for page in (cover, copyright_page, part, chapter, notes):
        book.add_item(page)
    book.toc = [
        epub.Link("03-part01.xhtml", "Part I: Your Mind", "p1"),
        epub.Link("04-chapter01.xhtml", "1: Thoughts Pass", "c1"),
        epub.Link("05-notes.xhtml", "Notes", "n"),
    ]
    book.spine = [cover, copyright_page, part, chapter, notes]
    book.add_item(epub.EpubNcx())
    path = tmp_path / "calm.epub"
    epub.write_epub(str(path), book)
    return path


def test_epub_chapters_follow_the_toc_and_keep_paragraphs(book_path):
    chapters = extract_text_from_epub(book_path, chapter_aware=True)

    assert [title for title, _ in chapters] == ["Part I: Your Mind", "1: Thoughts Pass"]
    part_text, chapter_text = chapters[0][1], chapters[1][1]
    # The part page keeps its epigraph but not its list of chapters.
    assert part_text == "“How we spend our days is how we spend our lives.”"
    assert chapter_text.split("\n\n") == [
        "It’s fine to have a thought.",
        "Past mistakes: We all screw up.",
        "Maybe you carry regret.",
        "1. Thoughts are not facts.",
    ]


def test_epub_metadata(book_path):
    metadata = extract_epub_metadata(book_path)
    assert (metadata["title"], metadata["author"], metadata["year"]) == (
        "Calm Mind", "Ann Writer", "2020",
    )


def test_normalize_keeps_apostrophes_and_reads_numerals():
    text = "Part II: You won’t — and floats/thoughts . . . pass (mostly) 2–5 times."
    assert _normalize_for_tts(text, spell_numbers=False) == (
        "Part Two: You won't, and floats or thoughts ... pass (mostly) 2 to 5 times."
    )
    assert "two to five" in _normalize_for_tts(text)
    assert _normalize_for_tts("This is terrifying.\ufeff Soft\u00adly.", spell_numbers=False) == (
        "This is terrifying. Softly."
    )
    assert _normalize_for_tts("It costs $42.50, about") == (
        "It costs forty-two dollars and fifty cents, about"
    )


def test_clean_language_replaces_whole_swear_words_only(tmp_path):
    text = "I call bullshit. (Holy shit!) Kick some ass, what the HELL. Hello, class, assume."
    assert _clean_language(text) == (
        "I call baloney. (Holy crap!) Kick some butt, what the HECK. Hello, class, assume."
    )
    converter = TTSConverter(output_dir=str(tmp_path), clean_language=True)
    assert converter._normalize("Oh, shit — it’s a damn mess.") == "Oh, crap, it's a darn mess."
    assert TTSConverter(output_dir=str(tmp_path))._normalize("Oh, shit.") == "Oh, shit."


def test_spoken_titles():
    assert _spoken_title("1: Thoughts Pass") == "Chapter 1. Thoughts Pass."
    assert _spoken_title("Part I: Your Mind") == "Part I. Your Mind."
    assert _spoken_title("Acknowledgments") == "Acknowledgments."


def test_chunks_never_cross_paragraphs(tmp_path):
    converter = TTSConverter(output_dir=str(tmp_path), chunk_size=40)
    plan = converter._chunk_plan("One short sentence. Another one here. A third.\n\nNew paragraph:")
    assert plan == [
        ("One short sentence. Another one here.", converter.SENTENCE_GAP),
        ("A third.", converter.PARAGRAPH_GAP),
        ("New paragraph.", converter.PARAGRAPH_GAP),
    ]


class _ToneWorker:
    """Stands in for the TTS server: 0.1 s of tone per word, padded with silence."""

    def __init__(self):
        self.texts = []

    def generate_batch(self, texts):
        self.texts += texts
        sr = 24000
        wavs = []
        for text in texts:
            tone = 0.3 * np.sin(np.arange(int(sr * 0.1 * len(text.split()))) / 5)
            wavs.append(np.concatenate([np.zeros(sr // 2), tone, np.zeros(sr // 2)]).astype(np.float32))
        return wavs, sr


def test_m4b_has_a_marker_per_chapter_and_tags(tmp_path):
    converter = TTSConverter(output_dir=str(tmp_path), output_format="m4b")
    converter.workers = [_ToneWorker()]
    chapters = [("Part I: Your Mind", "Opening words here."), ("1: Thoughts Pass", "A.\n\nB c.")]

    [path] = converter.convert_book_m4b(
        chapters, "calm", {"title": "Calm Mind", "author": "Ann Writer", "year": "2020"},
    )

    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout)
    titles = [c["tags"]["title"] for c in probe["chapters"]]
    assert titles == ["Opening Credits", "Part I: Your Mind", "1: Thoughts Pass", "End Credits"]
    assert probe["format"]["tags"]["artist"] == "Ann Writer"
    # Each chapter opens with its spoken title.
    spoken = converter.workers[0].texts
    assert spoken[0] == "Calm Mind. Written by Ann Writer."
    assert spoken[1:3] == ["Part One. Your Mind.", "Opening words here."]
    assert spoken[3] == "Chapter 1. Thoughts Pass."
    # The trimmed tones plus planned gaps: chapter 2 is title + 2 paragraphs.
    chapter = probe["chapters"][2]
    duration = float(chapter["end_time"]) - float(chapter["start_time"])
    expected = 0.4 + 0.06 + 1.2 + 0.1 + 0.06 + 0.75 + 0.2 + 0.06 + 2.0
    assert duration == pytest.approx(expected, abs=0.05)
