import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from google import genai
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# =========================================================
# CONFIG
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

MAX_COMPANIES = int(os.getenv("MAX_COMPANIES", "5"))
MAX_JOBS_OUTPUT = int(os.getenv("MAX_JOBS_OUTPUT", "500"))
MAX_JOB_LINKS_PER_COMPANY = int(os.getenv("MAX_JOB_LINKS_PER_COMPANY", "50"))
MAX_SECONDARY_PAGES_PER_COMPANY = int(os.getenv("MAX_SECONDARY_PAGES_PER_COMPANY", "6"))

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "45"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "1"))

MIN_JOB_DESCRIPTION_CHARS = int(os.getenv("MIN_JOB_DESCRIPTION_CHARS", "300"))
MAX_JOB_DESCRIPTION_CHARS = int(os.getenv("MAX_JOB_DESCRIPTION_CHARS", "60000"))

COMPANIES_CSV = Path("companies.csv")
COMPANIES_CAREER_PAGES_CSV = Path("companies_career_pages.csv")

BLACKLIST_CSV = Path("blacklist.csv")
CUSTOMERS_CSV = Path("customers.csv")
CHURNED_CSV = Path("churned.csv")
CURRENT_JOBS_CSV = Path("current jobs.csv")
IRRELEVANT_NAMES_CSV = Path("irrelevant names.csv")

OUTPUT_DIR = Path("output")
OUTPUT_CSV_PATH = OUTPUT_DIR / "gemini_company_jobs.csv"
RAW_JSON_PATH = OUTPUT_DIR / "raw_gemini_company_jobs.json"
SKIPPED_JOBS_CSV_PATH = OUTPUT_DIR / "skipped_jobs_debug.csv"

FIELDNAMES = [
    "List",
    "source",
    "company_domain",
    "company_name",
    "career_page_url",
    "job_title",
    "job_url",
    "job_location",
    "job_description",
    "job_description_length",
    "job_posted_date",
    "job_posted_date_source",
    "confidence",
    "reason",
    "checked_at_utc",
]

SKIPPED_FIELDNAMES = [
    "company_domain",
    "career_page_url",
    "seed_job_title",
    "seed_job_url",
    "final_job_url",
    "skip_reason",
    "job_page_text_sample",
    "checked_at_utc",
]

CAREER_LINK_KEYWORDS = [
    "careers",
    "career",
    "jobs",
    "join-us",
    "join us",
    "work-with-us",
    "work with us",
    "vacancies",
    "open roles",
    "opportunities",
    "hiring",
    "recruitment",
]

JOB_LINK_KEYWORDS = [
    "job",
    "jobs",
    "vacancy",
    "vacancies",
    "role",
    "roles",
    "position",
    "positions",
    "opening",
    "openings",
    "apply",
    "apply now",
    "view vacancy",
    "view vacancies",
    "view role",
    "view roles",
    "view job",
    "view jobs",
    "view details",
    "view details / apply",
    "details",
    "read more",
    "learn more",
]

NEXT_PAGE_KEYWORDS = [
    "open positions",
    "positions",
    "vacancies",
    "current vacancies",
    "view vacancies",
    "see vacancies",
    "see jobs",
    "view jobs",
    "job openings",
    "open roles",
    "browse roles",
    "search jobs",
    "find jobs",
    "opportunities",
    "join our team",
    "our vacancies",
    "explore roles",
    "all jobs",
    "learn more",
]

JOB_URL_PATTERNS = [
    "/jobs/",
    "/job/",
    "/vacancies/",
    "/vacancy/",
    "/roles/",
    "/role/",
    "/openings/",
    "/positions/",
    "/position/",
    "/careers/",
    "/career/",
    "/vacancyinformation",
    "greenhouse.io",
    "lever.co",
    "workable.com",
    "ashbyhq.com",
    "smartrecruiters.com",
    "pinpointhq.com",
    "bamboohr.com",
    "recruitee.com",
    "teamtailor.com",
    "workdayjobs.com",
    "myworkdayjobs.com",
    "icims.com",
    "jobvite.com",
    "personio.com",
    "oraclecloud.com",
    "kallidusrecruit.com",
]

ATS_HOST_PATTERNS = [
    "greenhouse.io",
    "lever.co",
    "workable.com",
    "ashbyhq.com",
    "smartrecruiters.com",
    "pinpointhq.com",
    "bamboohr.com",
    "recruitee.com",
    "teamtailor.com",
    "workdayjobs.com",
    "myworkdayjobs.com",
    "icims.com",
    "jobvite.com",
    "personio.com",
    "oraclecloud.com",
    "kallidusrecruit.com",
]

BLOCKED_EXTERNAL_JOB_BOARDS = [
    "builtin.com",
    "indeed.com",
    "glassdoor.com",
    "linkedin.com",
    "wellfound.com",
    "monster.com",
    "reed.co.uk",
    "totaljobs.com",
    "cv-library.co.uk",
    "ziprecruiter.com",
]

NON_JOB_LINK_KEYWORDS = [
    "privacy",
    "cookie",
    "terms",
    "contact",
    "about",
    "blog",
    "news",
    "login",
    "signin",
    "sign-in",
    "register",
    "facebook",
    "linkedin",
    "instagram",
    "twitter",
    "youtube",
]

CLOSED_OR_INVALID_JOB_SIGNALS = [
    "page not found",
    "404",
    "not found",
    "vacancy is no longer available",
    "job is no longer available",
    "position is no longer available",
    "this job is closed",
    "this vacancy has closed",
    "applications are closed",
    "no longer accepting applications",
    "this role is closed",
    "this position has been filled",
    "job has expired",
    "vacancy has expired",
    "expired vacancy",
    "sorry, this vacancy is no longer available",
    "the page you are looking for could not be found",
]

ZERO_JOB_RETRY_SIGNALS = [
    "vacancies",
    "open positions",
    "search jobs",
    "view jobs",
    "loading",
    "apply now",
    "open roles",
    "job openings",
    "join our team",
    "current vacancies",
    "learn more",
]

JOB_TITLE_HINTS = [
    "engineer",
    "developer",
    "manager",
    "executive",
    "assistant",
    "specialist",
    "consultant",
    "analyst",
    "coordinator",
    "designer",
    "administrator",
    "architect",
    "lead",
    "head of",
    "director",
    "technician",
    "associate",
    "officer",
    "advisor",
    "marketing",
    "sales",
    "finance",
    "account",
    "support",
    "product",
    "operations",
    "project",
    "data",
    "scientist",
]

ATTACHMENT_EXTENSIONS = [".pdf", ".doc", ".docx"]


# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_domain(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("https://", "").replace("http://", "")
    text = text.replace("www.", "")
    text = text.split("/")[0].strip()
    return text


def normalize_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rstrip("/")


def company_website(domain: str) -> str:
    domain = normalize_domain(domain)
    return f"https://{domain}"


def domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return normalize_domain(parsed.netloc)
    except Exception:
        return ""


def same_or_subdomain(url: str, company_domain: str) -> bool:
    try:
        host = normalize_domain(urlparse(url).netloc)
        domain = normalize_domain(company_domain)
        return host == domain or host.endswith("." + domain)
    except Exception:
        return False


def clean_page_text(text: str, max_chars: int = 70000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


def clean_job_description(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()

    noise_patterns = [
        r"(?i)cookie policy.*?(accept all|reject all|manage preferences)",
        r"(?i)we use cookies.*?(accept|reject|manage)",
        r"(?i)skip to content",
        r"(?i)share this job",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_JOB_DESCRIPTION_CHARS]


def is_closed_or_invalid_page(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True

    return any(signal in normalized for signal in CLOSED_OR_INVALID_JOB_SIGNALS)


def is_apply_only_page(text: str) -> bool:
    normalized = normalize_text(text)

    if not normalized:
        return True

    strong_apply_form_signals = [
        "apply for this job",
        "submit application",
        "attach your file",
        "attach your file(s)",
        "drag and drop",
        "maximum file size",
        "first name",
        "last name",
        "email address",
        "powered by",
        "report this job",
    ]

    matched = sum(1 for signal in strong_apply_form_signals if signal in normalized)

    real_description_signals = [
        "about the role",
        "the role",
        "responsibilities",
        "requirements",
        "about you",
        "what you will do",
        "what you'll do",
        "key responsibilities",
        "skills and experience",
        "experience required",
        "qualifications",
        "benefits",
    ]

    has_real_content = any(signal in normalized for signal in real_description_signals)

    if matched >= 4 and not has_real_content:
        return True

    return False


def has_real_job_description(text: str) -> bool:
    cleaned = clean_job_description(text)
    if len(cleaned) < MIN_JOB_DESCRIPTION_CHARS:
        return False

    if is_closed_or_invalid_page(cleaned):
        return False

    if is_apply_only_page(cleaned):
        return False

    role_signals = [
        "responsibilities",
        "requirements",
        "the role",
        "about the role",
        "job description",
        "what you will do",
        "what you'll do",
        "essential criteria",
        "key responsibilities",
        "skills",
        "experience",
        "about you",
        "qualifications",
        "benefits",
    ]

    normalized = normalize_text(cleaned)
    return any(signal in normalized for signal in role_signals)


def is_ats_url(url: str) -> bool:
    url_norm = normalize_text(url)
    return any(pattern in url_norm for pattern in ATS_HOST_PATTERNS)


def is_blocked_external_board(url: str) -> bool:
    try:
        host = normalize_domain(urlparse(url).netloc)
    except Exception:
        return False
    return any(host == domain or host.endswith("." + domain) for domain in BLOCKED_EXTERNAL_JOB_BOARDS)


def is_attachment_url(url: str) -> bool:
    url_norm = normalize_text(url)
    return any(url_norm.endswith(ext) or f"{ext}?" in url_norm for ext in ATTACHMENT_EXTENSIONS)


def is_likely_job_url(url: str, text: str = "", nearby_text: str = "") -> bool:
    combined = normalize_text(f"{url} {text} {nearby_text}")
    if any(bad in combined for bad in NON_JOB_LINK_KEYWORDS):
        return False

    if is_blocked_external_board(url):
        return False

    if any(pattern in combined for pattern in JOB_URL_PATTERNS):
        return True

    if any(keyword in combined for keyword in JOB_LINK_KEYWORDS):
        return True

    return False


def is_likely_next_page(url: str, text: str = "", nearby_text: str = "") -> bool:
    combined = normalize_text(f"{url} {text} {nearby_text}")

    if any(bad in combined for bad in NON_JOB_LINK_KEYWORDS):
        return False

    if is_blocked_external_board(url):
        return False

    if is_ats_url(url):
        return True

    if any(keyword in combined for keyword in NEXT_PAGE_KEYWORDS):
        return True

    return False


def has_zero_job_retry_signal(text: str) -> bool:
    normalized = normalize_text(text)
    return any(signal in normalized for signal in ZERO_JOB_RETRY_SIGNALS)


def title_similarity_score(a: str, b: str) -> int:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return 0
    if a_norm == b_norm:
        return 100
    if a_norm in b_norm or b_norm in a_norm:
        return 85

    a_tokens = set(re.findall(r"[a-z0-9]+", a_norm))
    b_tokens = set(re.findall(r"[a-z0-9]+", b_norm))
    if not a_tokens or not b_tokens:
        return 0

    overlap = len(a_tokens & b_tokens)
    if overlap == 0:
        return 0

    ratio = int((2 * overlap / (len(a_tokens) + len(b_tokens))) * 100)
    return ratio


def extract_headings_from_text(text: str) -> List[str]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    headings = []
    for line in lines:
        if not line:
            continue
        if len(line) > 120:
            continue
        if re.fullmatch(r"[A-Za-z0-9&,\-–—'()/+. ]{3,120}", line):
            headings.append(line)
    return headings[:12]


def looks_like_job_title(text: str) -> bool:
    value = normalize_text(text)
    if not value:
        return False
    if len(value) < 4 or len(value) > 120:
        return False
    if any(bad in value for bad in NON_JOB_LINK_KEYWORDS):
        return False
    return any(hint in value for hint in JOB_TITLE_HINTS)


def best_card_match_for_title(job_title: str, job_cards: List[Dict[str, str]], company_domain: str) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_score = 0
    title_norm = normalize_text(job_title)

    for card in job_cards:
        href = normalize_url(card.get("href", ""))
        if not href:
            continue
        if is_blocked_external_board(href):
            continue
        if not same_or_subdomain(href, company_domain) and not is_ats_url(href):
            continue

        link_text = str(card.get("link_text") or "")
        nearby_text = str(card.get("nearby_text") or "")
        card_title = str(card.get("card_title") or "")

        candidate_texts = [card_title, link_text]
        candidate_texts.extend(extract_headings_from_text(nearby_text))

        local_best = 0
        for candidate in candidate_texts:
            local_best = max(local_best, title_similarity_score(title_norm, candidate))

        if title_norm and title_norm in normalize_text(nearby_text):
            local_best = max(local_best, 92)

        if local_best > best_score:
            best_score = local_best
            best = {
                "href": href,
                "score": local_best,
                "card_title": card_title,
                "link_text": link_text,
                "nearby_text": nearby_text[:1200],
            }

    return best


def patch_missing_seed_urls(job_seeds: List[Dict[str, Any]], job_cards: List[Dict[str, str]], company_domain: str) -> List[Dict[str, Any]]:
    patched: List[Dict[str, Any]] = []

    for seed in job_seeds:
        seed_copy = dict(seed)
        current_url = normalize_url(seed_copy.get("job_url", ""))
        title = str(seed_copy.get("job_title") or "").strip()

        if current_url:
            patched.append(seed_copy)
            continue

        if not title:
            patched.append(seed_copy)
            continue

        best = best_card_match_for_title(title, job_cards, company_domain)
        if best and best.get("href") and int(best.get("score", 0)) >= 60:
            href = normalize_url(best["href"])
            if is_attachment_url(href):
                seed_copy["job_url"] = ""
                reason = str(seed_copy.get("reason") or "").strip()
                patch_note = "Matching vacancy appears to be attachment-only (PDF/DOC), not a normal HTML job page."
                seed_copy["reason"] = f"{reason} {patch_note}".strip()
            else:
                seed_copy["job_url"] = href
                reason = str(seed_copy.get("reason") or "").strip()
                patch_note = f"URL patched from page card match ({best.get('score')})."
                seed_copy["reason"] = f"{reason} {patch_note}".strip()

        patched.append(seed_copy)

    return patched


# =========================================================
# REFERENCE FILES
# =========================================================

def read_company_file(path: Path) -> tuple[Set[str], Set[str]]:
    domains = set()
    names = set()

    if not path.exists():
        print(f"[WARN] {path} not found. Skipping.")
        return domains, names

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            domain = normalize_domain(row.get("company_website"))
            name = normalize_text(row.get("company_name"))

            if domain:
                domains.add(domain)
            if name:
                names.add(name)

    print(f"[INFO] Loaded {path}: {len(domains)} domains, {len(names)} names")
    return domains, names


def read_current_jobs_file(path: Path) -> tuple[Set[str], Set[str], Set[str]]:
    domains = set()
    names = set()
    urls = set()

    if not path.exists():
        print(f"[WARN] {path} not found. Skipping.")
        return domains, names, urls

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            domain = normalize_domain(row.get("company_website"))
            name = normalize_text(row.get("company_name"))
            url = normalize_url(row.get("external_url"))

            if domain:
                domains.add(domain)
            if name:
                names.add(name)
            if url:
                urls.add(url)

    print(f"[INFO] Loaded {path}: {len(domains)} domains, {len(names)} names, {len(urls)} urls")
    return domains, names, urls


def read_irrelevant_keywords(path: Path) -> List[str]:
    keywords = []

    if not path.exists():
        print(f"[WARN] {path} not found. Skipping.")
        return keywords

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if "irrelevant_names" not in (reader.fieldnames or []):
            raise ValueError("irrelevant names.csv must have a header named irrelevant_names")

        for row in reader:
            keyword = normalize_text(row.get("irrelevant_names"))
            if keyword:
                keywords.append(keyword)

    print(f"[INFO] Loaded irrelevant keywords: {len(keywords)}")
    return keywords


def load_reference_data() -> Dict[str, Any]:
    blacklist_domains, blacklist_names = read_company_file(BLACKLIST_CSV)
    customers_domains, customers_names = read_company_file(CUSTOMERS_CSV)
    churned_domains, churned_names = read_company_file(CHURNED_CSV)
    active_domains, active_names, active_urls = read_current_jobs_file(CURRENT_JOBS_CSV)
    irrelevant_keywords = read_irrelevant_keywords(IRRELEVANT_NAMES_CSV)

    return {
        "blacklist_domains": blacklist_domains,
        "blacklist_names": blacklist_names,
        "customers_domains": customers_domains,
        "customers_names": customers_names,
        "churned_domains": churned_domains,
        "churned_names": churned_names,
        "active_domains": active_domains,
        "active_names": active_names,
        "active_urls": active_urls,
        "irrelevant_keywords": irrelevant_keywords,
    }


def read_company_domains() -> List[str]:
    if not COMPANIES_CSV.exists():
        return []

    domains = []

    with COMPANIES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if "company_domain" not in (reader.fieldnames or []):
            raise ValueError("companies.csv must have a header named company_domain")

        for row in reader:
            domain = normalize_domain(row.get("company_domain"))
            if domain:
                domains.append(domain)

    unique_domains = []
    seen = set()

    for domain in domains:
        if domain in seen:
            continue
        seen.add(domain)
        unique_domains.append(domain)

    if MAX_COMPANIES > 0:
        unique_domains = unique_domains[:MAX_COMPANIES]

    print(f"[INFO] Loaded company domains for this run: {len(unique_domains)}")
    return unique_domains


def read_direct_career_pages() -> List[Dict[str, str]]:
    if not COMPANIES_CAREER_PAGES_CSV.exists():
        print("[INFO] companies_career_pages.csv not found. Will use companies.csv instead.")
        return []

    pages = []

    with COMPANIES_CAREER_PAGES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if "career_page" not in (reader.fieldnames or []):
            raise ValueError("companies_career_pages.csv must have a header named career_page")

        for row in reader:
            career_page = normalize_url(row.get("career_page"))
            if not career_page:
                continue

            if not career_page.startswith("http"):
                career_page = "https://" + career_page

            company_domain = domain_from_url(career_page)

            if not company_domain:
                continue

            pages.append({
                "company_domain": company_domain,
                "career_page": career_page,
            })

    unique_pages = []
    seen = set()

    for item in pages:
        key = normalize_url(item["career_page"])
        if key in seen:
            continue
        seen.add(key)
        unique_pages.append(item)

    if MAX_COMPANIES > 0:
        unique_pages = unique_pages[:MAX_COMPANIES]

    print(f"[INFO] Loaded direct career pages for this run: {len(unique_pages)}")
    return unique_pages


# =========================================================
# FILTER HELPERS
# =========================================================

def company_matches(company_domain: str, company_name: str, domains: Set[str], names: Set[str]) -> bool:
    domain = normalize_domain(company_domain)
    name = normalize_text(company_name)

    if domain and domain in domains:
        return True

    if name and name in names:
        return True

    return False


def title_has_irrelevant_keyword(title: str, keywords: List[str]) -> bool:
    title_text = normalize_text(title)
    if not title_text:
        return False

    return any(keyword in title_text for keyword in keywords)


def assign_list_value(company_domain: str, company_name: str, ref: Dict[str, Any]) -> str:
    if company_matches(company_domain, company_name, ref["churned_domains"], ref["churned_names"]):
        return "churned"

    if company_matches(company_domain, company_name, ref["active_domains"], ref["active_names"]):
        return "active"

    return "inactive"


def should_remove_job(row: Dict[str, Any], ref: Dict[str, Any]) -> bool:
    company_domain = row.get("company_domain", "")
    company_name = row.get("company_name", "")
    title = row.get("job_title", "")
    job_url = normalize_url(row.get("job_url", ""))

    if company_matches(company_domain, company_name, ref["blacklist_domains"], ref["blacklist_names"]):
        return True

    if company_matches(company_domain, company_name, ref["customers_domains"], ref["customers_names"]):
        return True

    if title_has_irrelevant_keyword(title, ref["irrelevant_keywords"]):
        return True

    if job_url and job_url in ref["active_urls"]:
        return True

    return False


# =========================================================
# PLAYWRIGHT HELPERS
# =========================================================

def block_unneeded_requests(route):
    request = route.request
    resource_type = request.resource_type

    if resource_type in {"image", "font", "media"}:
        route.abort()
    else:
        route.continue_()


def safe_goto(page, url: str) -> bool:
    try:
        page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
        return True
    except PlaywrightTimeoutError:
        print(f"[WARN] Timeout opening {url}; trying domcontentloaded")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_SECONDS * 1000)
            return True
        except Exception as exc:
            print(f"[WARN] Failed opening {url}: {exc}")
            return False
    except Exception as exc:
        print(f"[WARN] Failed opening {url}: {exc}")
        return False


def rendered_page_text(page, max_chars: int = 70000) -> str:
    try:
        text = page.locator("body").inner_text(timeout=5000)
        return clean_page_text(text, max_chars=max_chars)
    except Exception:
        return ""


def rendered_html(page) -> str:
    try:
        return page.content()
    except Exception:
        return ""


def expand_page(page) -> None:
    try:
        for amount in (1200, 1800, 2500, 3500):
            page.mouse.wheel(0, amount)
            time.sleep(0.35)
    except Exception:
        pass

    click_texts = [
        "Open positions",
        "Vacancies",
        "View vacancies",
        "See vacancies",
        "View jobs",
        "See jobs",
        "Job openings",
        "Open roles",
        "View all jobs",
        "Current vacancies",
        "Join our team",
        "Search jobs",
        "Find jobs",
        "Open positions - Vacancies",
        "View Details / Apply",
    ]

    for text in click_texts:
        try:
            locator = page.get_by_text(text, exact=False).first
            if locator.count() > 0 and locator.is_visible(timeout=800):
                locator.click(timeout=1500)
                time.sleep(1)
        except Exception:
            continue


def extract_button_navigation_candidates(page, company_domain: str) -> List[str]:
    try:
        items = page.evaluate(
            """
            () => {
                const els = Array.from(document.querySelectorAll('button, [role="button"], a, div, span'));
                return els.map(el => {
                    const href = el.getAttribute('href') || '';
                    const dataHref = el.getAttribute('data-href') || '';
                    const dataUrl = el.getAttribute('data-url') || '';
                    const onclick = el.getAttribute('onclick') || '';
                    const text = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim();
                    return { href, dataHref, dataUrl, onclick, text };
                });
            }
            """
        )
    except Exception:
        items = []

    urls = []
    seen = set()

    url_pattern = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)

    for item in items:
        text = str(item.get("text") or "").strip()
        candidates = [
            item.get("href") or "",
            item.get("dataHref") or "",
            item.get("dataUrl") or "",
        ]

        onclick = str(item.get("onclick") or "")
        candidates.extend(url_pattern.findall(onclick))

        for raw_url in candidates:
            raw_url = str(raw_url or "").strip()
            if not raw_url:
                continue

            if raw_url.startswith("/"):
                raw_url = urljoin(page.url, raw_url)

            raw_url = normalize_url(raw_url)
            if not raw_url.startswith("http"):
                continue
            if is_blocked_external_board(raw_url):
                continue
            if not same_or_subdomain(raw_url, company_domain) and not is_ats_url(raw_url):
                continue
            if not is_likely_next_page(raw_url, text, onclick):
                continue
            if raw_url in seen:
                continue
            seen.add(raw_url)
            urls.append(raw_url)

    return urls[:30]


def extract_attachment_candidates(page, company_domain: str) -> List[Dict[str, str]]:
    cards = []
    seen = set()

    try:
        items = page.evaluate(
            """
            () => {
              const links = Array.from(document.querySelectorAll('a[href]'));
              return links.map(a => {
                let container = a;
                for (let i = 0; i < 5; i++) {
                  if (container.parentElement) container = container.parentElement;
                }

                const headings = Array.from(container.querySelectorAll('h1,h2,h3,h4,h5,h6,strong,b'))
                  .map(x => (x.innerText || '').trim())
                  .filter(Boolean)
                  .slice(0, 6);

                return {
                  href: a.href,
                  link_text: (a.innerText || a.getAttribute('aria-label') || a.getAttribute('title') || '').trim(),
                  nearby_text: (container.innerText || '').trim().slice(0, 2500),
                  headings
                };
              });
            }
            """
        )
    except Exception:
        items = []

    for item in items:
        href = normalize_url(item.get("href", ""))
        if not href:
            continue
        if is_blocked_external_board(href):
            continue
        if not same_or_subdomain(href, company_domain) and not is_ats_url(href):
            continue
        if not is_attachment_url(href):
            continue

        link_text = str(item.get("link_text") or "").strip()
        nearby_text = clean_page_text(item.get("nearby_text", ""), max_chars=2500)
        headings = item.get("headings") or []
        card_title = str(headings[0] if headings else "").strip()

        combined = normalize_text(f"{href} {link_text} {nearby_text} {card_title}")
        if not (
            "job" in combined
            or "vacanc" in combined
            or "role" in combined
            or looks_like_job_title(link_text)
            or looks_like_job_title(card_title)
        ):
            continue

        key = f"{href}|{card_title}|attachment"
        if key in seen:
            continue
        seen.add(key)

        cards.append({
            "href": href,
            "link_text": link_text,
            "nearby_text": nearby_text,
            "card_title": card_title,
            "source": "attachment",
        })

    return cards[:40]


def extract_clickthrough_cards(page, company_domain: str) -> List[Dict[str, str]]:
    cards: List[Dict[str, str]] = []
    seen = set()

    try:
        items = page.evaluate(
            """
            () => {
                const els = Array.from(document.querySelectorAll('a, button, [role="button"]'));
                return els.map((el, idx) => {
                    let container = el;
                    for (let i = 0; i < 5; i++) {
                        if (container.parentElement) container = container.parentElement;
                    }

                    const headings = Array.from(container.querySelectorAll('h1,h2,h3,h4,h5,h6,strong,b'))
                      .map(x => (x.innerText || '').trim())
                      .filter(Boolean)
                      .slice(0, 6);

                    return {
                        index: idx,
                        tag: el.tagName.toLowerCase(),
                        text: (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim(),
                        href: el.href || el.getAttribute('href') || '',
                        nearby_text: (container.innerText || '').trim().slice(0, 2500),
                        headings
                    };
                });
            }
            """
        )
    except Exception:
        items = []

    wanted_texts = [
        "learn more",
        "view vacancy",
        "view role",
        "view job",
        "view details",
        "view details / apply",
        "read more",
        "apply",
        "apply now",
        "details",
    ]

    for item in items[:120]:
        text = normalize_text(item.get("text", ""))
        href = str(item.get("href") or "").strip()
        nearby_text = clean_page_text(item.get("nearby_text", ""), max_chars=2500)
        headings = item.get("headings") or []
        card_title = str(headings[0] if headings else "").strip()

        if not text:
            continue

        if not any(normalize_text(t) in text for t in wanted_texts):
            continue

        if href:
            if href.startswith("/"):
                href = urljoin(page.url, href)
            href = normalize_url(href)
            if href.startswith("http") and not is_blocked_external_board(href):
                if same_or_subdomain(href, company_domain) or is_ats_url(href):
                    key = f"{href}|{card_title}|href"
                    if key not in seen:
                        seen.add(key)
                        cards.append({
                            "href": href,
                            "link_text": item.get("text", ""),
                            "nearby_text": nearby_text,
                            "card_title": card_title,
                            "source": "clickthrough_href",
                        })
            continue

        popup_page = None
        previous_url = normalize_url(page.url)

        def handle_popup(p):
            nonlocal popup_page
            popup_page = p

        try:
            page.once("popup", handle_popup)
            locator = page.locator("a, button, [role='button']").nth(int(item.get("index", 0)))
            if not locator.is_visible(timeout=400):
                continue
            locator.click(timeout=1200)
            time.sleep(1)
        except Exception:
            continue

        new_url = ""
        try:
            if popup_page:
                popup_page.wait_for_load_state("domcontentloaded", timeout=5000)
                new_url = normalize_url(popup_page.url)
            else:
                current_url = normalize_url(page.url)
                if current_url != previous_url:
                    new_url = current_url
        except Exception:
            new_url = ""

        if new_url and new_url.startswith("http") and not is_blocked_external_board(new_url):
            if same_or_subdomain(new_url, company_domain) or is_ats_url(new_url):
                key = f"{new_url}|{card_title}|clicked"
                if key not in seen:
                    seen.add(key)
                    cards.append({
                        "href": new_url,
                        "link_text": item.get("text", ""),
                        "nearby_text": nearby_text,
                        "card_title": card_title,
                        "source": "clickthrough_clicked",
                    })

        try:
            if popup_page:
                popup_page.close()
            else:
                current_url = normalize_url(page.url)
                if current_url != previous_url:
                    page.go_back(wait_until="domcontentloaded", timeout=5000)
                    time.sleep(0.5)
        except Exception:
            pass

    return cards[:40]


def extract_nearby_job_cards(page, company_domain: str) -> List[Dict[str, str]]:
    cards = []
    seen = set()

    try:
        anchor_cards = page.evaluate(
            """
            () => {
              const links = Array.from(document.querySelectorAll('a[href]'));
              return links.map(a => {
                let container = a;
                for (let i = 0; i < 6; i++) {
                  if (container.parentElement) container = container.parentElement;
                }

                const headings = Array.from(container.querySelectorAll('h1,h2,h3,h4,h5,h6,strong,b'))
                  .map(x => (x.innerText || '').trim())
                  .filter(Boolean)
                  .slice(0, 6);

                return {
                  href: a.href,
                  link_text: (a.innerText || a.getAttribute('aria-label') || a.getAttribute('title') || '').trim(),
                  nearby_text: (container.innerText || '').trim().slice(0, 2500),
                  headings
                };
              });
            }
            """
        )
    except Exception:
        anchor_cards = []

    for card in anchor_cards:
        href = normalize_url(card.get("href", ""))
        nearby_text = clean_page_text(card.get("nearby_text", ""), max_chars=2500)
        link_text = str(card.get("link_text") or "").strip()
        headings = card.get("headings") or []
        card_title = str(headings[0] if headings else "").strip()

        if not href or not nearby_text:
            continue
        if is_blocked_external_board(href):
            continue
        if not same_or_subdomain(href, company_domain) and not is_ats_url(href):
            continue
        if is_attachment_url(href):
            continue
        if not is_likely_job_url(href, link_text, nearby_text):
            continue

        key = f"{href}|{card_title}|{nearby_text[:120]}"
        if key in seen:
            continue
        seen.add(key)

        cards.append({
            "href": href,
            "link_text": link_text,
            "nearby_text": nearby_text,
            "card_title": card_title,
            "source": "anchor",
        })

    page_text_sample = rendered_page_text(page, max_chars=12000)

    for href in extract_button_navigation_candidates(page, company_domain):
        key = f"{href}|button-nav"
        if key in seen:
            continue
        seen.add(key)
        cards.append({
            "href": href,
            "link_text": "button-navigation",
            "nearby_text": page_text_sample[:1800],
            "card_title": "",
            "source": "button_navigation",
        })

    for card in extract_clickthrough_cards(page, company_domain):
        href = normalize_url(card.get("href", ""))
        card_title = str(card.get("card_title") or "").strip()
        key = f"{href}|{card_title}|clickthrough"
        if key in seen:
            continue
        seen.add(key)
        cards.append(card)

    for card in extract_attachment_candidates(page, company_domain):
        href = normalize_url(card.get("href", ""))
        card_title = str(card.get("card_title") or "").strip()
        key = f"{href}|{card_title}|attachment"
        if key in seen:
            continue
        seen.add(key)
        cards.append(card)

    return cards[:260]


def extract_next_page_candidates(page, company_domain: str) -> List[str]:
    try:
        links = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('a[href], button')).map(el => {
                const href = el.tagName.toLowerCase() === 'a' ? el.href : '';
                return {
                    text: (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim(),
                    href: href,
                    outer_html: el.outerHTML.slice(0, 700)
                };
            })
            """
        )
    except Exception:
        links = []

    candidates = []
    seen = set()

    for item in links:
        href = normalize_url(item.get("href", ""))
        text = str(item.get("text") or "").strip()
        outer_html = str(item.get("outer_html") or "").strip()

        if not href or not href.startswith("http"):
            continue
        if is_blocked_external_board(href):
            continue
        if is_attachment_url(href):
            continue
        if not same_or_subdomain(href, company_domain) and not is_ats_url(href):
            continue

        combined_extra = f"{text} {outer_html}"

        if not is_likely_next_page(href, text, combined_extra):
            continue

        if href in seen:
            continue
        seen.add(href)
        candidates.append(href)

    for href in extract_button_navigation_candidates(page, company_domain):
        if href not in seen:
            seen.add(href)
            candidates.append(href)

    for card in extract_clickthrough_cards(page, company_domain):
        href = normalize_url(card.get("href", ""))
        if href and href not in seen and not is_attachment_url(href):
            seen.add(href)
            candidates.append(href)

    return candidates[:40]


def find_career_pages_from_domain(context, domain: str) -> List[str]:
    home_url = company_website(domain)
    page = context.new_page()
    page.route("**/*", block_unneeded_requests)

    career_urls = []

    if safe_goto(page, home_url):
        expand_page(page)
        candidates = extract_next_page_candidates(page, domain)

        for href in candidates:
            combined = normalize_text(href)
            if (any(keyword in combined for keyword in CAREER_LINK_KEYWORDS) or is_ats_url(href)) and not is_blocked_external_board(href):
                career_urls.append(href)

    page.close()

    fallback_paths = [
        "/careers",
        "/career",
        "/jobs",
        "/join-us",
        "/join",
        "/vacancies",
        "/work-with-us",
        "/opportunities",
        "/recruitment",
    ]

    for path in fallback_paths:
        career_urls.append(company_website(domain) + path)

    unique = []
    seen = set()

    for url in career_urls:
        url = normalize_url(url)
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)

    return unique[:15]


# =========================================================
# STRUCTURED DATA / POSTED DATE
# =========================================================

def find_dates_in_object(obj: Any, source_prefix: str = "structured_data") -> List[Dict[str, str]]:
    results = []

    date_keys = {
        "dateposted",
        "datepublished",
        "posteddate",
        "createdat",
        "published_at",
        "datecreated",
        "uploaddate",
    }

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = normalize_text(key).replace(" ", "").replace("_", "")
            if key_norm in date_keys and value:
                results.append({
                    "date": str(value),
                    "source": f"{source_prefix}_{key}",
                })
            else:
                results.extend(find_dates_in_object(value, source_prefix))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_dates_in_object(item, source_prefix))

    return results


def extract_structured_dates_from_html(html: str) -> List[Dict[str, str]]:
    dates = []

    soup = BeautifulSoup(html or "", "html.parser")

    for script in soup.find_all("script", type=lambda x: x and "ld+json" in x.lower()):
        raw = script.string or script.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        dates.extend(find_dates_in_object(data, "structured_data"))

    meta_names = [
        "article:published_time",
        "datePublished",
        "date",
        "publish_date",
        "published_time",
        "dateposted",
        "datePosted",
    ]

    normalized_meta_names = {normalize_text(x) for x in meta_names}

    for meta in soup.find_all("meta"):
        name = meta.get("name") or meta.get("property") or ""
        content = meta.get("content") or ""
        if not name or not content:
            continue

        if normalize_text(name) in normalized_meta_names:
            dates.append({
                "date": str(content),
                "source": f"meta_{name}",
            })

    return dates


def first_structured_date(html: str) -> tuple[str, str]:
    dates = extract_structured_dates_from_html(html)
    if not dates:
        return "", ""

    return dates[0]["date"], dates[0]["source"]


# =========================================================
# PAGE-LEVEL SIGNAL HELPERS
# =========================================================

def extract_page_level_titles(page_text: str) -> List[str]:
    titles: List[str] = []
    seen: Set[str] = set()

    for line in str(page_text or "").splitlines():
        value = re.sub(r"\s+", " ", line).strip()
        if not value:
            continue
        if len(value) > 120:
            continue
        if looks_like_job_title(value):
            key = normalize_text(value)
            if key not in seen:
                seen.add(key)
                titles.append(value)

    return titles[:20]


def infer_page_level_skip_reason(page_text: str, job_cards: List[Dict[str, str]]) -> str:
    text_norm = normalize_text(page_text)
    attachments = [c for c in job_cards if c.get("source") == "attachment"]
    empty_or_attachment_cards = [
        c for c in job_cards
        if not normalize_url(c.get("href", "")) or is_attachment_url(normalize_url(c.get("href", "")))
    ]

    if attachments:
        return "Vacancy appears to exist on page, but only attachment-based job files (PDF/DOC) were found."
    if any("apply only" in normalize_text(c.get("nearby_text", "")) for c in job_cards):
        return "Vacancy signals found, but only apply-only / non-descriptive pages were detected."
    if "current vacancies" in text_norm or "job openings" in text_norm or "open positions" in text_norm:
        return "Career page shows vacancy signals, but no usable job seeds were extracted."
    if empty_or_attachment_cards:
        return "Job-like blocks were found, but no normal HTML job detail URLs were extracted."
    return "Career page shows likely open-job signals, but no usable job seeds were extracted."


# =========================================================
# GEMINI
# =========================================================

def build_career_page_prompt(company_domain: str, career_page_url: str, page_text: str, job_cards: List[Dict[str, str]]) -> str:
    cards_text = json.dumps(job_cards[:220], ensure_ascii=False, indent=2)

    return f"""
You are extracting currently open job vacancies from a rendered company career page.

Company domain: {company_domain}
Career page URL: {career_page_url}

Rules:
- Return ONLY valid JSON.
- No markdown.
- Output must be an object with key "jobs".
- "jobs" must be a list.
- Extract ALL currently open jobs visible in the provided rendered career page content.
- Do not filter by category.
- Do not invent jobs.
- Do not include closed, expired, unavailable, speculative, or generic talent pool roles unless clearly listed as open.
- If there are no currently open jobs, return {{"jobs":[]}}.
- Prefer the most specific normal HTML job URL from JOB_CARDS when available.
- JOB_CARDS may contain the exact job detail page URL even when the main page only shows a generic button like "View Details / Apply".
- If a job title in the page text matches a JOB_CARD nearby_text/card_title, use that JOB_CARD href as job_url.
- If a visible job exists but only a PDF/DOC attachment is available, still return the job with job_url = "" and mention attachment-only in reason.
- If a visible job exists directly on the career page and there is no separate normal job URL, still return the job with job_url = "" and explain that the job is only on the career page / no separate detail URL.
- Do not use unrelated external job-board URLs.
- Do not use the career page URL as job_url if a more specific vacancy/job URL is not available.
- If the exact job detail URL is not available, use an empty string.
- confidence should be "high", "medium", or "low".
- reason should be short and explain why the role was extracted or why no normal URL exists.

Return this JSON shape:
{{
  "jobs": [
    {{
      "company_name": "",
      "job_title": "",
      "job_url": "",
      "job_location": "",
      "confidence": "",
      "reason": ""
    }}
  ]
}}

RENDERED_CAREER_PAGE_TEXT:
{page_text}

JOB_CARDS_WITH_LINKS:
{cards_text}
""".strip()


def build_job_page_prompt(
    company_domain: str,
    job_url: str,
    job_page_text: str,
    structured_posted_date: str,
    structured_posted_date_source: str,
) -> str:
    return f"""
You are extracting structured data from an open job page.

Company domain: {company_domain}
Job URL: {job_url}

Structured posted date found by parser: {structured_posted_date}
Structured posted date source: {structured_posted_date_source}

Rules:
- Return ONLY valid JSON.
- No markdown.
- Do not invent information.
- Extract only what is explicitly present in the job page text or structured posted date fields.
- If the page says page not found, job not found, vacancy unavailable, expired, closed, apply-only, or does not contain a real job description, return:
  {{"is_valid_open_job": false}}
- Otherwise return:
  {{"is_valid_open_job": true, ...fields...}}
- job_posted_date:
  - Use the structured posted date if provided.
  - Otherwise only extract a visible posted/published date if explicitly shown in the job page text.
  - If no posted date is available, return empty string.
- job_posted_date_source:
  - If using structured posted date, return the provided structured source.
  - If using visible text, return "visible_text".
  - If no date exists, return empty string.
- Do NOT summarize job_description.
- The script will save the full job description itself, so you do not need to return job_description.
- confidence should be "high", "medium", or "low".
- reason should be short.

Return this JSON shape:
{{
  "is_valid_open_job": true,
  "company_name": "",
  "job_title": "",
  "job_url": "",
  "job_location": "",
  "job_posted_date": "",
  "job_posted_date_source": "",
  "confidence": "",
  "reason": ""
}}

JOB_PAGE_TEXT:
{job_page_text}
""".strip()


def parse_gemini_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        return {}

    return data


def extract_jobs_from_career_page_with_gemini(
    client: genai.Client,
    company_domain: str,
    career_page_url: str,
    page_text: str,
    job_cards: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    prompt = build_career_page_prompt(company_domain, career_page_url, page_text, job_cards)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    data = parse_gemini_json(response.text or "")
    jobs = data.get("jobs", [])

    if not isinstance(jobs, list):
        return []

    jobs = [job for job in jobs if isinstance(job, dict)]
    jobs = patch_missing_seed_urls(jobs, job_cards, company_domain)
    return jobs


def enrich_job_page_with_gemini(
    client: genai.Client,
    company_domain: str,
    job_url: str,
    job_page_text: str,
    structured_posted_date: str,
    structured_posted_date_source: str,
) -> Dict[str, Any]:
    prompt = build_job_page_prompt(
        company_domain=company_domain,
        job_url=job_url,
        job_page_text=job_page_text[:70000],
        structured_posted_date=structured_posted_date,
        structured_posted_date_source=structured_posted_date_source,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    data = parse_gemini_json(response.text or "")

    if not isinstance(data, dict):
        return {}

    return data


# =========================================================
# JOB PAGE PROCESSING
# =========================================================

def open_job_page_and_extract(context, client: genai.Client, company_domain: str, job_seed: Dict[str, Any]) -> Dict[str, Any]:
    seed_url = normalize_url(job_seed.get("job_url"))

    if not seed_url:
        reason = str(job_seed.get("reason") or "").strip()
        if "attachment" in normalize_text(reason) or "pdf" in normalize_text(reason) or "doc" in normalize_text(reason):
            return {
                "is_valid_open_job": False,
                "invalid_reason": "Visible vacancy found, but only PDF/DOC attachment exists and no normal HTML job page is available.",
            }
        return {
            "is_valid_open_job": False,
            "invalid_reason": "Visible vacancy found, but no specific normal job URL was available.",
        }

    if is_attachment_url(seed_url):
        return {
            "is_valid_open_job": False,
            "invalid_reason": "Visible vacancy found, but only PDF/DOC attachment exists and no normal HTML job page is available.",
            "job_url": seed_url,
        }

    page = context.new_page()
    page.route("**/*", block_unneeded_requests)

    final_url = seed_url
    page_text = ""
    html = ""

    try:
        if safe_goto(page, seed_url):
            final_url = normalize_url(page.url)
            expand_page(page)
            page_text = rendered_page_text(page, max_chars=90000)
            html = rendered_html(page)
    finally:
        page.close()

    structured_date, structured_source = first_structured_date(html)
    cleaned_description = clean_job_description(page_text)

    if is_apply_only_page(cleaned_description):
        return {
            "is_valid_open_job": False,
            "invalid_reason": "Job page is only an apply form and does not contain a real job description.",
            "job_url": final_url or seed_url,
            "job_page_text_sample": page_text[:1000],
        }

    if not has_real_job_description(cleaned_description):
        return {
            "is_valid_open_job": False,
            "invalid_reason": "Job page is closed, invalid, not found, or has no real job description.",
            "job_url": final_url or seed_url,
            "job_page_text_sample": page_text[:1000],
        }

    try:
        enriched = enrich_job_page_with_gemini(
            client=client,
            company_domain=company_domain,
            job_url=final_url or seed_url,
            job_page_text=cleaned_description,
            structured_posted_date=structured_date,
            structured_posted_date_source=structured_source,
        )
    except Exception as exc:
        print(f"[WARN] Gemini job-page enrichment failed for {seed_url}: {exc}")
        enriched = {}

    if enriched.get("is_valid_open_job") is False:
        return {
            "is_valid_open_job": False,
            "invalid_reason": "Gemini classified job page as invalid/closed.",
            "job_url": final_url or seed_url,
        }

    return {
        "is_valid_open_job": True,
        "company_name": str(enriched.get("company_name") or job_seed.get("company_name") or "").strip(),
        "job_title": str(enriched.get("job_title") or job_seed.get("job_title") or "").strip(),
        "job_url": normalize_url(enriched.get("job_url") or final_url or seed_url),
        "job_location": str(enriched.get("job_location") or job_seed.get("job_location") or "").strip(),
        "job_description": cleaned_description,
        "job_description_length": len(cleaned_description),
        "job_posted_date": str(enriched.get("job_posted_date") or structured_date or "").strip(),
        "job_posted_date_source": str(enriched.get("job_posted_date_source") or structured_source or "").strip(),
        "confidence": str(enriched.get("confidence") or job_seed.get("confidence") or "medium").strip(),
        "reason": str(enriched.get("reason") or job_seed.get("reason") or "").strip(),
    }


# =========================================================
# CAREER PAGE PROCESSING
# =========================================================

def add_skipped_job(
    skipped_jobs: List[Dict[str, Any]],
    company_domain: str,
    career_page_url: str,
    seed: Dict[str, Any],
    reason: str,
    checked_at: str,
    final_job_url: str = "",
    job_page_text_sample: str = "",
) -> None:
    skipped_jobs.append({
        "company_domain": company_domain,
        "career_page_url": career_page_url,
        "seed_job_title": str(seed.get("job_title") or ""),
        "seed_job_url": str(seed.get("job_url") or ""),
        "final_job_url": final_job_url,
        "skip_reason": reason,
        "job_page_text_sample": job_page_text_sample,
        "checked_at_utc": checked_at,
    })


def add_page_level_skipped_jobs(
    skipped_jobs: List[Dict[str, Any]],
    company_domain: str,
    career_page_url: str,
    page_text: str,
    titles: List[str],
    reason: str,
    checked_at: str,
) -> None:
    if titles:
        for title in titles[:12]:
            skipped_jobs.append({
                "company_domain": company_domain,
                "career_page_url": career_page_url,
                "seed_job_title": title,
                "seed_job_url": "",
                "final_job_url": "",
                "skip_reason": reason,
                "job_page_text_sample": clean_page_text(page_text, max_chars=1000),
                "checked_at_utc": checked_at,
            })
    else:
        skipped_jobs.append({
            "company_domain": company_domain,
            "career_page_url": career_page_url,
            "seed_job_title": "",
            "seed_job_url": "",
            "final_job_url": "",
            "skip_reason": reason,
            "job_page_text_sample": clean_page_text(page_text, max_chars=1000),
            "checked_at_utc": checked_at,
        })


def process_single_rendered_page(
    context,
    client: genai.Client,
    company_domain: str,
    root_career_page_url: str,
    page_url: str,
    ref: Dict[str, Any],
    checked_at: str,
    skipped_jobs_debug: List[Dict[str, Any]],
    force_deeper: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    rows = []

    raw = {
        "company_domain": company_domain,
        "career_page_url": root_career_page_url,
        "processed_page_url": page_url,
        "job_seeds": [],
        "jobs": [],
        "skipped_jobs": [],
        "next_pages": [],
        "error": "",
        "force_deeper": force_deeper,
    }

    page = context.new_page()
    page.route("**/*", block_unneeded_requests)

    page_text = ""
    job_cards: List[Dict[str, str]] = []
    next_pages: List[str] = []

    try:
        ok = safe_goto(page, page_url)
        if not ok:
            raw["error"] = "Could not open page."
            return rows, raw, []

        expand_page(page)
        if force_deeper:
            expand_page(page)
            time.sleep(0.8)

        page_text = rendered_page_text(page, max_chars=90000)
        job_cards = extract_nearby_job_cards(page, company_domain)
        next_pages = extract_next_page_candidates(page, company_domain)

        raw["career_page_text_sample"] = page_text[:5000]
        raw["job_cards_sample"] = job_cards[:60]
        raw["next_pages"] = next_pages[:20]

        job_seeds = extract_jobs_from_career_page_with_gemini(
            client=client,
            company_domain=company_domain,
            career_page_url=page_url,
            page_text=page_text,
            job_cards=job_cards,
        )

        job_seeds = [
            seed for seed in job_seeds
            if not is_blocked_external_board(str(seed.get("job_url") or ""))
        ]

        job_seeds = patch_missing_seed_urls(job_seeds, job_cards, company_domain)

        raw["job_seeds"] = job_seeds

    except Exception as exc:
        raw["error"] = str(exc)
        print(f"[ERROR] Failed page {page_url}: {exc}")
        return rows, raw, []
    finally:
        page.close()

    # page-level skip logging when there are strong signals but zero usable seeds
    if not job_seeds:
        page_titles = extract_page_level_titles(page_text)
        attachments = [c for c in job_cards if c.get("source") == "attachment"]

        strong_signal = bool(page_titles) or bool(attachments) or has_zero_job_retry_signal(page_text)

        if strong_signal:
            reason = infer_page_level_skip_reason(page_text, job_cards)
            add_page_level_skipped_jobs(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                page_text=page_text,
                titles=page_titles,
                reason=reason,
                checked_at=checked_at,
            )
            raw["skipped_jobs"].append({
                "seed": {},
                "reason": reason,
                "job_url": "",
                "job_page_text_sample": page_text[:1000],
                "page_level_titles": page_titles[:12],
            })

    job_seeds = job_seeds[:MAX_JOB_LINKS_PER_COMPANY]

    for seed in job_seeds:
        job_data = open_job_page_and_extract(context, client, company_domain, seed)

        if not job_data.get("is_valid_open_job"):
            reason = job_data.get("invalid_reason", "Invalid or closed job page.")
            final_job_url = job_data.get("job_url", seed.get("job_url", ""))
            sample = job_data.get("job_page_text_sample", "")

            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": reason,
                "job_url": final_job_url,
                "job_page_text_sample": sample,
            })

            add_skipped_job(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                seed=seed,
                reason=reason,
                checked_at=checked_at,
                final_job_url=final_job_url,
                job_page_text_sample=sample,
            )
            continue

        job_title = str(job_data.get("job_title") or "").strip()
        if not job_title:
            reason = "Missing job title after job page extraction."
            final_job_url = job_data.get("job_url", seed.get("job_url", ""))

            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": reason,
                "job_url": final_job_url,
            })

            add_skipped_job(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                seed=seed,
                reason=reason,
                checked_at=checked_at,
                final_job_url=final_job_url,
            )
            continue

        company_name = str(job_data.get("company_name") or "").strip()
        job_url = normalize_url(job_data.get("job_url") or seed.get("job_url") or "")

        if not job_url:
            reason = "Missing job URL."

            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": reason,
            })

            add_skipped_job(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                seed=seed,
                reason=reason,
                checked_at=checked_at,
            )
            continue

        if is_blocked_external_board(job_url):
            reason = "Removed because URL belongs to blocked external job board."

            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": reason,
                "job_url": job_url,
            })

            add_skipped_job(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                seed=seed,
                reason=reason,
                checked_at=checked_at,
                final_job_url=job_url,
            )
            continue

        row = {
            "List": "",
            "source": "playwright + gemini",
            "company_domain": company_domain,
            "company_name": company_name,
            "career_page_url": root_career_page_url,
            "job_title": job_title,
            "job_url": job_url,
            "job_location": str(job_data.get("job_location") or "").strip(),
            "job_description": str(job_data.get("job_description") or "").strip(),
            "job_description_length": int(job_data.get("job_description_length") or len(str(job_data.get("job_description") or ""))),
            "job_posted_date": str(job_data.get("job_posted_date") or "").strip(),
            "job_posted_date_source": str(job_data.get("job_posted_date_source") or "").strip(),
            "confidence": str(job_data.get("confidence") or "").strip(),
            "reason": str(job_data.get("reason") or "").strip(),
            "checked_at_utc": checked_at,
        }

        if not has_real_job_description(row["job_description"]):
            reason = "Final row removed because job_description is missing/too short/invalid."

            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": reason,
                "job_url": job_url,
            })

            add_skipped_job(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                seed=seed,
                reason=reason,
                checked_at=checked_at,
                final_job_url=job_url,
                job_page_text_sample=row["job_description"][:1000],
            )
            continue

        if should_remove_job(row, ref):
            reason = "Removed by blacklist/customers/current jobs/irrelevant names logic."

            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": reason,
                "job_url": job_url,
            })

            add_skipped_job(
                skipped_jobs=skipped_jobs_debug,
                company_domain=company_domain,
                career_page_url=root_career_page_url,
                seed=seed,
                reason=reason,
                checked_at=checked_at,
                final_job_url=job_url,
                job_page_text_sample=row["job_description"][:1000],
            )
            continue

        row["List"] = assign_list_value(company_domain, company_name, ref)

        rows.append(row)
        raw["jobs"].append(row)

        if len(rows) >= MAX_JOBS_OUTPUT:
            break

        time.sleep(SLEEP_SECONDS)

    return rows, raw, next_pages


def process_career_page(
    context,
    client: genai.Client,
    company_domain: str,
    career_page_url: str,
    ref: Dict[str, Any],
    checked_at: str,
    skipped_jobs_debug: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows = []
    raw_items = []

    queue = [normalize_url(career_page_url)]
    visited = set()

    while queue and len(visited) < MAX_SECONDARY_PAGES_PER_COMPANY:
        current_page_url = queue.pop(0)
        if not current_page_url or current_page_url in visited:
            continue

        visited.add(current_page_url)

        rows, raw, next_pages = process_single_rendered_page(
            context=context,
            client=client,
            company_domain=company_domain,
            root_career_page_url=career_page_url,
            page_url=current_page_url,
            ref=ref,
            checked_at=checked_at,
            skipped_jobs_debug=skipped_jobs_debug,
            force_deeper=False,
        )

        raw_items.append(raw)
        all_rows.extend(rows)

        if rows:
            break

        current_text_sample = raw.get("career_page_text_sample", "")
        should_retry_deeper = has_zero_job_retry_signal(current_text_sample) or len(next_pages) > 0

        if should_retry_deeper:
            retry_rows, retry_raw, retry_next_pages = process_single_rendered_page(
                context=context,
                client=client,
                company_domain=company_domain,
                root_career_page_url=career_page_url,
                page_url=current_page_url,
                ref=ref,
                checked_at=checked_at,
                skipped_jobs_debug=skipped_jobs_debug,
                force_deeper=True,
            )

            retry_raw["retry_reason"] = "Zero jobs found on first pass; stronger retry executed."
            raw_items.append(retry_raw)
            all_rows.extend(retry_rows)

            if retry_rows:
                break

            for next_page in retry_next_pages:
                next_page = normalize_url(next_page)
                if next_page and next_page not in visited and next_page not in queue:
                    queue.append(next_page)

        for next_page in next_pages:
            next_page = normalize_url(next_page)
            if next_page and next_page not in visited and next_page not in queue:
                queue.append(next_page)

        time.sleep(SLEEP_SECONDS)

    return all_rows, raw_items


def process_domain_discovery(
    context,
    client: genai.Client,
    domain: str,
    ref: Dict[str, Any],
    checked_at: str,
    skipped_jobs_debug: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    career_urls = find_career_pages_from_domain(context, domain)

    all_rows = []
    raw_items = []

    for career_url in career_urls[:5]:
        rows, raws = process_career_page(
            context=context,
            client=client,
            company_domain=domain,
            career_page_url=career_url,
            ref=ref,
            checked_at=checked_at,
            skipped_jobs_debug=skipped_jobs_debug,
        )

        all_rows.extend(rows)
        raw_items.extend(raws)

        if all_rows:
            break

        time.sleep(SLEEP_SECONDS)

    return all_rows, raw_items


# =========================================================
# OUTPUT
# =========================================================

def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []

    for row in rows:
        key = normalize_url(row.get("job_url")) or f"{normalize_domain(row.get('company_domain'))}|{normalize_text(row.get('job_title'))}"

        if key in seen:
            continue

        seen.add(key)
        output.append(row)

    return output


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it as a GitHub Secret.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=GEMINI_API_KEY)
    ref = load_reference_data()
    checked_at = datetime.now(timezone.utc).isoformat()

    all_rows = []
    raw_results = []
    skipped_jobs_debug = []

    direct_pages = read_direct_career_pages()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
        )

        if direct_pages:
            print("[INFO] Using companies_career_pages.csv direct career page mode.")

            for index, item in enumerate(direct_pages, start=1):
                if len(all_rows) >= MAX_JOBS_OUTPUT:
                    break

                domain = item["company_domain"]
                career_page = item["career_page"]

                print(f"\n[INFO] Career page {index}/{len(direct_pages)}: {career_page}")

                rows, raws = process_career_page(
                    context=context,
                    client=client,
                    company_domain=domain,
                    career_page_url=career_page,
                    ref=ref,
                    checked_at=checked_at,
                    skipped_jobs_debug=skipped_jobs_debug,
                )

                all_rows.extend(rows)
                raw_results.extend(raws)

                time.sleep(SLEEP_SECONDS)

        else:
            print("[INFO] Using companies.csv domain discovery mode.")
            domains = read_company_domains()

            for index, domain in enumerate(domains, start=1):
                if len(all_rows) >= MAX_JOBS_OUTPUT:
                    break

                print(f"\n[INFO] Company {index}/{len(domains)}: {domain}")

                rows, raw_items = process_domain_discovery(
                    context=context,
                    client=client,
                    domain=domain,
                    ref=ref,
                    checked_at=checked_at,
                    skipped_jobs_debug=skipped_jobs_debug,
                )

                all_rows.extend(rows)
                raw_results.extend(raw_items)

                time.sleep(SLEEP_SECONDS)

        context.close()
        browser.close()

    all_rows = dedupe_rows(all_rows)
    all_rows = all_rows[:MAX_JOBS_OUTPUT]

    write_csv(str(OUTPUT_CSV_PATH), all_rows, FIELDNAMES)
    write_csv(str(SKIPPED_JOBS_CSV_PATH), skipped_jobs_debug, SKIPPED_FIELDNAMES)
    write_json(RAW_JSON_PATH, raw_results)

    print("\n[DONE]")
    print(f"Jobs saved: {len(all_rows)} -> {OUTPUT_CSV_PATH}")
    print(f"Skipped jobs debug saved: {len(skipped_jobs_debug)} -> {SKIPPED_JOBS_CSV_PATH}")
    print(f"Raw JSON -> {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
