import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from google import genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

MAX_COMPANIES = int(os.getenv("MAX_COMPANIES", "5"))
MAX_JOBS_OUTPUT = int(os.getenv("MAX_JOBS_OUTPUT", "500"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "1"))

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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


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


def clean_page_text(text: str, max_chars: int = 30000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_chars]


def same_or_subdomain(url: str, company_domain: str) -> bool:
    try:
        host = normalize_domain(urlparse(url).netloc)
        domain = normalize_domain(company_domain)
        return host == domain or host.endswith("." + domain)
    except Exception:
        return False


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
    """
    Reads companies_career_pages.csv if it exists.

    Expected format:
    career_page
    https://example.com/careers
    https://example.com/jobs
    """
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


def fetch_html(url: str) -> Optional[str]:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )

        print(f"[INFO] GET {url} -> {response.status_code}")

        if response.status_code >= 400:
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None

        return response.text

    except Exception as exc:
        print(f"[WARN] Failed to fetch {url}: {exc}")
        return None


def extract_links(base_url: str, html: str, company_domain: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = str(a.get("href") or "").strip()
        text = normalize_text(a.get_text(" "))
        full_url = urljoin(base_url, href)

        if not full_url.startswith("http"):
            continue

        combined = normalize_text(f"{href} {text}")

        if any(keyword in combined for keyword in CAREER_LINK_KEYWORDS):
            links.append(full_url)

    unique = []
    seen = set()

    for link in links:
        link = normalize_url(link)
        if link in seen:
            continue
        seen.add(link)
        unique.append(link)

    return unique[:15]


def fallback_career_urls(domain: str) -> List[str]:
    base = company_website(domain)

    paths = [
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

    return [base + path for path in paths]


def find_career_pages(domain: str) -> List[str]:
    home_url = company_website(domain)
    html = fetch_html(home_url)

    candidates = []

    if html:
        candidates.extend(extract_links(home_url, html, domain))

    candidates.extend(fallback_career_urls(domain))

    unique = []
    seen = set()

    for url in candidates:
        url = normalize_url(url)
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)

    return unique[:15]


def html_to_page_text(html: str, max_chars: int = 30000) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    return clean_page_text(soup.get_text(" "), max_chars=max_chars)


def collect_career_page_text_from_direct_url(domain: str, career_page_url: str) -> tuple[List[Dict[str, str]], List[str]]:
    pages = []
    successful_urls = []

    html = fetch_html(career_page_url)
    if not html:
        return pages, successful_urls

    text = html_to_page_text(html, max_chars=35000)

    if len(text) >= 150:
        pages.append({
            "url": career_page_url,
            "text": text,
        })
        successful_urls.append(career_page_url)

    return pages, successful_urls


def collect_career_page_text_from_domain(domain: str) -> tuple[List[Dict[str, str]], List[str]]:
    career_urls = find_career_pages(domain)
    pages = []
    successful_urls = []

    for url in career_urls:
        html = fetch_html(url)
        if not html:
            continue

        text = html_to_page_text(html, max_chars=30000)

        if len(text) < 150:
            continue

        pages.append({
            "url": url,
            "text": text,
        })
        successful_urls.append(url)

        if len(pages) >= 5:
            break

        time.sleep(SLEEP_SECONDS)

    return pages, successful_urls


def build_prompt(company_domain: str, pages: List[Dict[str, str]]) -> str:
    pages_text = "\n\n".join(
        f"CAREER_PAGE_URL: {page['url']}\nPAGE_TEXT:\n{page['text']}"
        for page in pages
    )

    return f"""
You are extracting open jobs from company career pages.

Company domain: {company_domain}

Rules:
- Return ONLY valid JSON.
- No markdown.
- Output must be an object with key "jobs".
- "jobs" must be a list.
- Extract ALL currently open jobs from the provided career page text.
- Do not filter by category.
- Do not invent jobs.
- Do not include closed, expired, unavailable, speculative, or generic talent pool roles unless clearly listed as open.
- If there are no currently open jobs, return {{"jobs":[]}}.
- If job URL is missing, use the career page URL where the job was found.
- confidence should be "high", "medium", or "low".
- reason should be short.

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

Career page content:
{pages_text}
""".strip()


def parse_gemini_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

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
        return {"jobs": []}

    if "jobs" not in data or not isinstance(data["jobs"], list):
        return {"jobs": []}

    return data


def extract_jobs_with_gemini(client: genai.Client, company_domain: str, pages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not pages:
        return []

    prompt = build_prompt(company_domain, pages)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    text = response.text or ""
    data = parse_gemini_json(text)

    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        return []

    return [job for job in jobs if isinstance(job, dict)]


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


def process_company(
    client: genai.Client,
    domain: str,
    pages: List[Dict[str, str]],
    successful_urls: List[str],
    ref: Dict[str, Any],
    checked_at: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []

    company_raw = {
        "company_domain": domain,
        "career_pages_checked": successful_urls,
        "jobs": [],
        "error": "",
    }

    try:
        jobs = extract_jobs_with_gemini(client, domain, pages)
    except Exception as exc:
        print(f"[ERROR] Gemini failed for {domain}: {exc}")
        company_raw["error"] = str(exc)
        return rows, company_raw

    for job in jobs:
        company_name = str(job.get("company_name") or "").strip()
        job_title = str(job.get("job_title") or "").strip()
        job_url = normalize_url(job.get("job_url"))
        job_location = str(job.get("job_location") or "").strip()
        confidence = str(job.get("confidence") or "").strip()
        reason = str(job.get("reason") or "").strip()

        if not job_title:
            continue

        if not job_url and successful_urls:
            job_url = successful_urls[0]

        row = {
            "List": "",
            "source": "gemini direct career page" if COMPANIES_CAREER_PAGES_CSV.exists() else "gemini company website",
            "company_domain": domain,
            "company_name": company_name,
            "career_page_url": successful_urls[0] if successful_urls else "",
            "job_title": job_title,
            "job_url": job_url,
            "job_location": job_location,
            "confidence": confidence,
            "reason": reason,
            "checked_at_utc": checked_at,
        }

        if should_remove_job(row, ref):
            continue

        row["List"] = assign_list_value(domain, company_name, ref)

        rows.append(row)
        company_raw["jobs"].append(row)

    return rows, company_raw


def main() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it as a GitHub Secret.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=GEMINI_API_KEY)
    ref = load_reference_data()
    checked_at = datetime.now(timezone.utc).isoformat()

    direct_pages = read_direct_career_pages()
    all_rows = []
    raw_results = []

    if direct_pages:
        print("[INFO] Using companies_career_pages.csv direct career page mode.")

        for index, item in enumerate(direct_pages, start=1):
            domain = item["company_domain"]
            career_page = item["career_page"]

            print(f"\n[INFO] Career page {index}/{len(direct_pages)}: {career_page}")

            pages, successful_urls = collect_career_page_text_from_direct_url(domain, career_page)

            rows, company_raw = process_company(
                client=client,
                domain=domain,
                pages=pages,
                successful_urls=successful_urls,
                ref=ref,
                checked_at=checked_at,
            )

            all_rows.extend(rows)
            raw_results.append(company_raw)

            if len(all_rows) >= MAX_JOBS_OUTPUT:
                break

            time.sleep(SLEEP_SECONDS)

    else:
        print("[INFO] Using companies.csv domain discovery mode.")

        domains = read_company_domains()

        for index, domain in enumerate(domains, start=1):
            print(f"\n[INFO] Company {index}/{len(domains)}: {domain}")

            pages, successful_urls = collect_career_page_text_from_domain(domain)

            rows, company_raw = process_company(
                client=client,
                domain=domain,
                pages=pages,
                successful_urls=successful_urls,
                ref=ref,
                checked_at=checked_at,
            )

            all_rows.extend(rows)
            raw_results.append(company_raw)

            if len(all_rows) >= MAX_JOBS_OUTPUT:
                break

            time.sleep(SLEEP_SECONDS)

    all_rows = dedupe_rows(all_rows)
    all_rows = all_rows[:MAX_JOBS_OUTPUT]

    write_csv(OUTPUT_CSV_PATH, all_rows)
    write_json(RAW_JSON_PATH, raw_results)

    print("\n[DONE]")
    print(f"Jobs saved: {len(all_rows)} -> {OUTPUT_CSV_PATH}")
    print(f"Raw JSON -> {RAW_JSON_PATH}")


if __name__ == "__main__":
    main()
