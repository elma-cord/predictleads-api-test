import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

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
    "job_posted_date",
    "job_posted_date_source",
    "confidence",
    "reason",
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
    "view vacancy",
    "view role",
    "view job",
    "read more",
    "learn more",
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

    # Remove common cookie/navigation noise, but keep the full role content.
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


def has_real_job_description(text: str) -> bool:
    cleaned = clean_job_description(text)
    if len(cleaned) < MIN_JOB_DESCRIPTION_CHARS:
        return False

    if is_closed_or_invalid_page(cleaned):
        return False

    # Require at least some role-like content.
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
        "apply",
    ]

    normalized = normalize_text(cleaned)
    return any(signal in normalized for signal in role_signals)


def safe_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


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


def extract_nearby_job_cards(page) -> List[Dict[str, str]]:
    """
    Extract visible link/card context from rendered page.
    This helps Gemini match job titles to real URLs.
    """
    try:
        cards = page.evaluate(
            """
            () => {
              const links = Array.from(document.querySelectorAll('a[href]'));
              return links.map(a => {
                let container = a;
                for (let i = 0; i < 6; i++) {
                  if (container.parentElement) container = container.parentElement;
                }
                return {
                  href: a.href,
                  link_text: (a.innerText || a.getAttribute('aria-label') || a.getAttribute('title') || '').trim(),
                  nearby_text: (container.innerText || '').trim().slice(0, 1500)
                };
              });
            }
            """
        )
    except Exception:
        cards = []

    cleaned = []
    seen = set()

    for card in cards:
        href = normalize_url(card.get("href", ""))
        nearby_text = clean_page_text(card.get("nearby_text", ""), max_chars=1500)
        link_text = str(card.get("link_text") or "").strip()

        if not href or not nearby_text:
            continue

        combined = normalize_text(f"{href} {link_text} {nearby_text}")
        if any(bad in combined for bad in NON_JOB_LINK_KEYWORDS):
            continue

        key = f"{href}|{nearby_text[:100]}"
        if key in seen:
            continue
        seen.add(key)

        cleaned.append({
            "href": href,
            "link_text": link_text,
            "nearby_text": nearby_text,
        })

    return cleaned[:120]


def extract_rendered_links(page, company_domain: str, job_like_only: bool = False) -> List[Dict[str, str]]:
    try:
        links = page.evaluate(
            """
            () => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                text: (a.innerText || a.getAttribute('aria-label') || a.getAttribute('title') || '').trim(),
                href: a.href,
                outer_html: a.outerHTML.slice(0, 500)
            }))
            """
        )
    except Exception:
        links = []

    output = []
    seen = set()

    for item in links:
        href = normalize_url(item.get("href", ""))
        text = str(item.get("text") or "").strip()
        outer_html = str(item.get("outer_html") or "").strip()

        if not href or not href.startswith("http"):
            continue

        if not same_or_subdomain(href, company_domain):
            continue

        combined = normalize_text(f"{href} {text} {outer_html}")

        if any(bad in combined for bad in NON_JOB_LINK_KEYWORDS):
            continue

        if job_like_only and not any(keyword in combined for keyword in JOB_LINK_KEYWORDS):
            continue

        if href in seen:
            continue

        seen.add(href)

        output.append({
            "text": text,
            "href": href,
            "outer_html": outer_html,
        })

    return output


def find_career_pages_from_domain(context, domain: str) -> List[str]:
    home_url = company_website(domain)
    page = context.new_page()
    page.route("**/*", block_unneeded_requests)

    career_urls = []

    if safe_goto(page, home_url):
        links = extract_rendered_links(page, domain, job_like_only=False)

        for link in links:
            combined = normalize_text(f"{link.get('href')} {link.get('text')}")
            if any(keyword in combined for keyword in CAREER_LINK_KEYWORDS):
                career_urls.append(link["href"])

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
# GEMINI
# =========================================================

def build_career_page_prompt(company_domain: str, career_page_url: str, page_text: str, job_cards: List[Dict[str, str]]) -> str:
    cards_text = json.dumps(job_cards[:120], ensure_ascii=False, indent=2)

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
- Use the most specific job URL from JOB_CARDS when available.
- Do not use the career page URL as job_url if a more specific vacancy/job URL is available.
- If the job URL is not visible/available, use an empty string.
- confidence should be "high", "medium", or "low".
- reason should be short and explain why the role was extracted.

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
- If the page says page not found, job not found, vacancy unavailable, expired, closed, or does not contain a real job description, return:
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
- job_description should be the full cleaned job description text from the job page, preserving the main sections as much as possible.
- Remove navigation/header/footer/cookie text if obvious.
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

    return [job for job in jobs if isinstance(job, dict)]


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
        return {
            "is_valid_open_job": False,
            "invalid_reason": "No specific job URL found.",
        }

    page = context.new_page()
    page.route("**/*", block_unneeded_requests)

    final_url = seed_url
    page_text = ""
    html = ""

    try:
        if safe_goto(page, seed_url):
            final_url = normalize_url(page.url)
            page_text = rendered_page_text(page, max_chars=90000)
            html = rendered_html(page)
    finally:
        page.close()

    structured_date, structured_source = first_structured_date(html)
    cleaned_description = clean_job_description(page_text)

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
        # Full description comes from Playwright page text, not from Gemini summary.
        "job_description": cleaned_description,
        "job_posted_date": str(enriched.get("job_posted_date") or structured_date or "").strip(),
        "job_posted_date_source": str(enriched.get("job_posted_date_source") or structured_source or "").strip(),
        "confidence": str(enriched.get("confidence") or job_seed.get("confidence") or "medium").strip(),
        "reason": str(enriched.get("reason") or job_seed.get("reason") or "").strip(),
    }


# =========================================================
# CAREER PAGE PROCESSING
# =========================================================

def process_career_page(
    context,
    client: genai.Client,
    company_domain: str,
    career_page_url: str,
    ref: Dict[str, Any],
    checked_at: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []

    raw = {
        "company_domain": company_domain,
        "career_page_url": career_page_url,
        "job_seeds": [],
        "jobs": [],
        "skipped_jobs": [],
        "error": "",
    }

    page = context.new_page()
    page.route("**/*", block_unneeded_requests)

    try:
        ok = safe_goto(page, career_page_url)
        if not ok:
            raw["error"] = "Could not open career page."
            return rows, raw

        page_text = rendered_page_text(page, max_chars=90000)
        job_cards = extract_nearby_job_cards(page)

        raw["career_page_text_sample"] = page_text[:5000]
        raw["job_cards_sample"] = job_cards[:30]

        job_seeds = extract_jobs_from_career_page_with_gemini(
            client=client,
            company_domain=company_domain,
            career_page_url=career_page_url,
            page_text=page_text,
            job_cards=job_cards,
        )

        raw["job_seeds"] = job_seeds

    except Exception as exc:
        raw["error"] = str(exc)
        print(f"[ERROR] Failed career page {career_page_url}: {exc}")
        return rows, raw
    finally:
        page.close()

    job_seeds = job_seeds[:MAX_JOB_LINKS_PER_COMPANY]

    for seed in job_seeds:
        job_data = open_job_page_and_extract(context, client, company_domain, seed)

        if not job_data.get("is_valid_open_job"):
            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": job_data.get("invalid_reason", "Invalid or closed job page."),
                "job_url": job_data.get("job_url", seed.get("job_url", "")),
            })
            continue

        job_title = str(job_data.get("job_title") or "").strip()
        if not job_title:
            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": "Missing job title after job page extraction.",
                "job_url": job_data.get("job_url", seed.get("job_url", "")),
            })
            continue

        company_name = str(job_data.get("company_name") or "").strip()
        job_url = normalize_url(job_data.get("job_url") or seed.get("job_url") or "")

        if not job_url:
            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": "Missing job URL.",
            })
            continue

        row = {
            "List": "",
            "source": "playwright + gemini",
            "company_domain": company_domain,
            "company_name": company_name,
            "career_page_url": career_page_url,
            "job_title": job_title,
            "job_url": job_url,
            "job_location": str(job_data.get("job_location") or "").strip(),
            "job_description": str(job_data.get("job_description") or "").strip(),
            "job_posted_date": str(job_data.get("job_posted_date") or "").strip(),
            "job_posted_date_source": str(job_data.get("job_posted_date_source") or "").strip(),
            "confidence": str(job_data.get("confidence") or "").strip(),
            "reason": str(job_data.get("reason") or "").strip(),
            "checked_at_utc": checked_at,
        }

        if not has_real_job_description(row["job_description"]):
            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": "Final row removed because job_description is missing/too short/invalid.",
                "job_url": job_url,
            })
            continue

        if should_remove_job(row, ref):
            raw["skipped_jobs"].append({
                "seed": seed,
                "reason": "Removed by blacklist/customers/current jobs/irrelevant names logic.",
                "job_url": job_url,
            })
            continue

        row["List"] = assign_list_value(company_domain, company_name, ref)

        rows.append(row)
        raw["jobs"].append(row)

        if len(rows) >= MAX_JOBS_OUTPUT:
            break

        time.sleep(SLEEP_SECONDS)

    return rows, raw


def process_domain_discovery(
    context,
    client: genai.Client,
    domain: str,
    ref: Dict[str, Any],
    checked_at: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    career_urls = find_career_pages_from_domain(context, domain)

    all_rows = []
    raw_items = []

    for career_url in career_urls[:5]:
        rows, raw = process_career_page(
            context=context,
            client=client,
            company_domain=domain,
            career_page_url=career_url,
            ref=ref,
            checked_at=checked_at,
        )

        all_rows.extend(rows)
        raw_items.append(raw)

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


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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

                rows, raw = process_career_page(
                    context=context,
                    client=client,
                    company_domain=domain,
                    career_page_url=career_page,
                    ref=ref,
                    checked_at=checked_at,
                )

                all_rows.extend(rows)
                raw_results.append(raw)

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
                )

                all_rows.extend(rows)
                raw_results.extend(raw_items)

                time.sleep(SLEEP_SECONDS)

        context.close()
        browser.close()

    all_rows = dedupe_rows(all_rows)
    all_rows = all_rows[:MAX_JOBS_OUTPUT]

    write_csv(OUTPUT_CSV_PATH, all_rows)
    write_json(RAW_JSON_PATH, raw_results)

    print("\n[DONE]")
    print(f"Jobs saved: {len(all_rows)} -> {OUTPUT_CSV_PATH}")
    print(f"Raw JSON -> {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
