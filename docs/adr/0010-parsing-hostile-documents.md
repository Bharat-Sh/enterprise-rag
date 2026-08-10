# ADR-0010 — Parsing hostile documents

- **Status:** Accepted
- **Date:** 2026-08-10
- **Milestone:** M3b

## Context

Until now every byte this system processed came from itself: credentials it
minted, rows it wrote, text it decoded. M3b is the first code that takes an
arbitrary binary file from an authenticated but otherwise untrusted user and
runs a format parser over it.

That is a different security posture from anything before it. A PDF is an object
graph with indirect references and can be self-referential; a DOCX is a ZIP
archive of XML, which means it arrives carrying two of the oldest attacks in the
book — a decompression bomb in the container and entity expansion in the payload
— and **both are enabled by default in the obvious implementation**.

## Decision

Parse in the worker, never in the API. Bound every dimension an attacker
controls, in the parser rather than only in a timeout. Choose libraries for
license and surface area, not speed.

### PDF: `pypdf`

| Rejected | Why |
| --- | --- |
| **PyMuPDF** | Fastest by a wide margin and the usual recommendation. **AGPL**, which would reach this entire codebase — the repository is MIT and intended to be published. A licensing constraint, not an engineering one, and not negotiable. |
| **pdfplumber** | Better layout reconstruction, which we would genuinely like. Built on `pdfminer.six`: markedly slower, and a larger surface for the same text-extraction job. Revisit if multi-column extraction becomes a measured quality problem. |

Bounded by a **page cap**, because a thousand-page scan is both a legitimate
document and an excellent way to occupy a worker for an hour. Truncation is
recorded in `metadata` rather than silent.

Encrypted PDFs are refused, after first trying the empty password — that case is
ubiquitous (it is how a tool marks a document read-only) and readable without
any secret at all, so refusing it outright would reject a large slice of
ordinary business documents.

### DOCX: `zipfile` + `defusedxml`, and no DOCX library

The obvious choice is `python-docx`, and it was rejected for a specific reason:
it hands the XML to `lxml` with a default parser, which **expands internal
entities**, and it does no decompression accounting whatsoever. Using it would
mean writing the ZIP hardening anyway and then either patching its parser or
trusting it.

Since the hardening is the substantial part and pulling text out of `w:t`
elements is not, the library would buy a dependency — plus `lxml`, a C extension
— while moving the security decisions out of view. Both stand-ins are standard
library or a single-purpose shim, and every limit is visible in one file.

**What that gives up:** styles, headers, footers, comments, and footnotes.
Headers and footers are usually a page number and a confidentiality banner
repeated on every page, which is noise that would otherwise be embedded into
every chunk — losing them is closer to a feature. Footnotes are a real loss and
are recorded as one.

**What it keeps:** table text, which is where most factual content in a business
document lives. The extractor walks the whole tree rather than the top level,
because table paragraphs are nested inside `w:tbl` and a shallow scan would
silently drop every table in the document.

### The limits, and what each one is for

| Limit | Attack it closes |
| --- | --- |
| `max_pdf_pages` | A vast document occupying a worker indefinitely. |
| `max_extracted_bytes` | A decompression bomb. The *upload* limit does nothing here — the compressed file is a few kilobytes. |
| `max_compression_ratio` | The same bomb, caught per entry. Ordinary office documents sit under 20:1; a bomb is thousands to one. |
| `max_archive_entries` | An archive of a million tiny files, which exhausts time and memory without tripping any size limit. |
| `defusedxml` | Billion-laughs and XXE. XXE is the serious one: a parser that resolves it returns `/etc/passwd` as document text, which is then chunked, embedded, and made **searchable**. |
| `word/document.xml` must exist | The magic number identifies a ZIP, not a DOCX. An XLSX or a JAR reaches the parser with identical bytes at the front. |

The archive checks run against the **central directory**, which the file
declares up front — so a bomb is rejected on its own stated numbers without
decompressing anything. The bounded read afterwards is what catches an archive
that lies about them, since that metadata is written by the attacker.

## The limitation we are not pretending away

`parse_timeout_seconds` wraps the parse in `anyio.fail_after` around a thread.
**Python cannot cancel a thread.** So a parse that genuinely hangs will fail its
job on schedule and leave a thread burning CPU until it finishes on its own —
which, for a true infinite loop, is never.

The caps above are what make that rare rather than routine: they are the primary
defence, and the timeout is the backstop. The complete fix is a process pool,
where a hung worker can actually be killed, and it is deferred rather than
forgotten — it costs inter-process serialisation of every document body, which
is real, and one job per worker process already bounds the blast radius to one
job and one container that an orchestrator will restart.

## Consequences

**We get:** five formats ingestible, with each attack class closed by an
explicit limit rather than by a library's defaults, and a test for each that
pins the *mechanism* — the entity tests assert `EntitiesForbidden` specifically,
so removing `defusedxml` fails them rather than passing on an incidental parse
error.

**We pay:**

- No OCR, so a scanned PDF fails as "no extractable text". Loud is right: the
  alternative is a `READY` document with zero chunks, permanently unfindable
  while appearing to have worked. OCR is a service with a GPU budget.
- No layout reconstruction, so multi-column PDFs extract in content-stream
  order, which is sometimes wrong.
- Hand-written DOCX extraction to maintain, against a format we do not control.
- A hung parse leaks a thread, as above.

## Alternatives considered

| Option | Assessment |
| --- | --- |
| **`unstructured`** | Handles every format with one interface. An enormous dependency tree — some paths pull in torch — slow installs, and an abstraction that hides exactly the per-format decisions this ADR is about. |
| **Parse in a subprocess per document** | The correct answer to the thread-cancellation problem, and the one to reach for when a hang is observed rather than hypothesised. |
| **Convert everything via LibreOffice** | Handles far more formats, and puts a large C++ binary with its own CVE history on the untrusted-input path. |
| **Trust the declared content type** | Would remove the sniffer. The oldest trick in the book is a file that lies about what it is. |
