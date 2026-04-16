import csv
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import requests


BASE_URL = "https://predictleads.com/api/v3"

API_KEY = os.getenv("PREDICTLEADS_API_KEY", "").strip()
API_TOKEN = os.getenv("PREDICTLEADS_API_TOKEN", "").strip()

HOURS_BACK = int(os.getenv("HOURS_BACK", "24"))

COMPANY_PER_PAGE = int(os.getenv("COMPANY_PER_PAGE", "100"))
GLOBAL_PER_PAGE = int(os.getenv("GLOBAL_PER_PAGE", "100"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

GLOBAL_LOCATION = os.getenv("GLOBAL_LOCATION", "United Kingdom").strip()
GLOBAL_ONET_CODES = os.getenv("GLOBAL_ONET_CODES", "15-1252.00").strip()
COMPANY_LOCATION_FILTER = os.getenv("COMPANY_LOCATION_FILTER", "United Kingdom").strip()

MAX_COMPANIES = int(os.getenv("MAX_COMPANIES", "0"))  # 0 = all companies
MAX_COMPANY_PAGES = int(os.getenv("MAX_COMPANY_PAGES", "10"))
MAX_GLOBAL_PAGES = int(os.getenv("MAX_GLOBAL_PAGES", "10"))

MAX_COMPANY_JOBS = int(os.getenv("MAX_COMPANY_JOBS", "1000"))
MAX_GLOBAL_JOBS = int(os.getenv("MAX_GLOBAL_JOBS", "1000"))
MAX_OUTPUT_ROWS = int(os.getenv("MAX_OUTPUT_ROWS", "2000"))

FETCH_COMPANY_JOBS = os.getenv("FETCH_COMPANY_JOBS", "true").lower() == "true"
FETCH_GLOBAL_JOBS = os.getenv("FETCH_GLOBAL_JOBS", "true").lower() == "true"

COMPANIES_CSV = Path("companies.csv")
BLACKLIST_CSV = Path("blacklist.csv")
CUSTOMERS_CSV = Path("customers.csv")
CHURNED_CSV = Path("churned.csv")
CURRENT_JOBS_CSV = Path("current jobs.csv")
IRRELEVANT_NAMES_CSV = Path("irrelevant names.csv")

OUTPUT_DIR = Path("output")

COMPANY_CSV_PATH = OUTPUT_DIR / "company_jobs.csv"
GLOBAL_CSV_PATH = OUTPUT_DIR / "global_jobs.csv"
COMBINED_CSV_PATH = OUTPUT_DIR / "combined_predictleads_jobs.csv"

RAW_COMPANY_JSON_PATH = OUTPUT_DIR / "raw_company_jobs.json"
RAW_GLOBAL_JSON_PATH = OUTPUT_DIR / "raw_global_jobs.json"


FIELDNAMES = [
    "List",
    "source", "source_company_domain", "global_location", "global_onet_codes",
    "company_location_filter", "id", "type", "title", "translated_title",
    "normalized_title", "description", "url", "first_seen_at", "last_seen_at",
    "last_processed_at", "posted_at", "contract_types", "categories",
    "onet_code", "onet_family", "onet_occupation_name",
    "recruiter_name", "recruiter_title", "recruiter_contact",
    "salary", "salary_low", "salary_high", "salary_currency",
    "salary_low_usd", "salary_high_usd", "salary_time_unit",
    "seniority", "status", "language", "location", "location_data",
    "tags", "company_id", "company_name", "company_domain",
    "company_ticker", "raw_json",
]


def safe_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_domain(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("https://", "").replace("http://", "")
    text = text.replace("www.", "")
    text = text.split("/")[0].strip()
    return text


def normalize_url(value: Any) -> str:
    text = normalize_text(value)
    text = text.rstrip("/")
    return text


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


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


def company_matches(row: Dict[str, Any], domains: Set[str], names: Set[str]) -> bool:
    company_domain = normalize_domain(row.get("company_domain"))
    source_company_domain = normalize_domain(row.get("source_company_domain"))
    company_name = normalize_text(row.get("company_name"))

    if company_domain and company_domain in domains:
        return True
    if source_company_domain and source_company_domain in domains:
        return True
    if company_name and company_name in names:
        return True

    return False


def title_has_irrelevant_keyword(title: Any, keywords: List[str]) -> bool:
    title_text = normalize_text(title)
    if not title_text:
        return False

    return any(keyword in title_text for keyword in keywords)


def enrich_and_filter_rows(rows: List[Dict[str, Any]], ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    filtered = []

    for row in rows:
        pl_url = normalize_url(row.get("url"))

        if company_matches(row, ref["blacklist_domains"], ref["blacklist_names"]):
            continue

        if company_matches(row, ref["customers_domains"], ref["customers_names"]):
            continue

        if title_has_irrelevant_keyword(row.get("title"), ref["irrelevant_keywords"]):
            continue

        if pl_url and pl_url in ref["active_urls"]:
            continue

        if company_matches(row, ref["churned_domains"], ref["churned_names"]):
            row["List"] = "churned"
        elif company_matches(row, ref["active_domains"], ref["active_names"]):
            row["List"] = "active"
        else:
            row["List"] = "inactive"

        filtered.append(row)

    return filtered


def location_matches(attrs: Dict[str, Any], required_location: str) -> bool:
    if not required_location:
        return True

    required = required_location.lower()

    location_text = str(attrs.get("location") or "").lower()
    if required in location_text:
        return True

    location_data = attrs.get("location_data") or []
    if isinstance(location_data, list):
        for item in location_data:
            if not isinstance(item, dict):
                continue

            values = [
                item.get("city"),
                item.get("state"),
                item.get("country"),
                item.get("region"),
                item.get("continent"),
            ]
            joined = " ".join(str(v).lower() for v in values if v)

            if required in joined:
                return True

    return False


def is_valid_predictleads_job(attrs: Dict[str, Any], cutoff: datetime) -> bool:
    if normalize_text(attrs.get("status")) == "closed":
        return False

    if attrs.get("language") != "en":
        return False

    if "united kingdom" not in normalize_text(attrs.get("location")):
        return False

    last_seen_at = parse_dt(attrs.get("last_seen_at"))
    if not last_seen_at or last_seen_at < cutoff:
        return False

    return True


def build_params(
    page: int,
    per_page: int,
    location: str = "",
    onet_codes: str = "",
) -> Dict[str, Any]:
    params = {
        "api_key": API_KEY,
        "api_token": API_TOKEN,
        "page": page,
        "per_page": per_page,
    }

    if location:
        params["location"] = location

    if onet_codes:
        params["onet_codes"] = onet_codes

    return params


def request_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    clean_params = {k: v for k, v in params.items() if v not in ["", None]}

    response = requests.get(
        url,
        params=clean_params,
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    print(f"[INFO] GET {response.url}")
    print(f"[INFO] Status: {response.status_code}")

    if response.status_code >= 400:
        print("[ERROR] PredictLeads request failed:")
        print(response.text[:3000])
        response.raise_for_status()

    return response.json()


def read_company_domains() -> List[str]:
    if not COMPANIES_CSV.exists():
        print("[WARN] companies.csv not found. Company-specific fetch will be skipped.")
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

    unique_domains = sorted(set(domains))

    if MAX_COMPANIES > 0:
        unique_domains = unique_domains[:MAX_COMPANIES]

    print(f"[INFO] Loaded company domains for this run: {len(unique_domains)}")
    return unique_domains


def company_lookup(included: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup = {}

    for item in included or []:
        if item.get("type") != "company":
            continue

        company_id = item.get("id")
        attrs = item.get("attributes", {}) or {}

        if company_id:
            lookup[company_id] = {
                "company_id": company_id,
                "company_domain": attrs.get("domain"),
                "company_name": attrs.get("company_name"),
                "company_ticker": attrs.get("ticker"),
            }

    return lookup


def flatten_job(
    job: Dict[str, Any],
    companies: Dict[str, Dict[str, Any]],
    source: str,
    source_company_domain: str = "",
    global_location: str = "",
    global_onet_codes: str = "",
    company_location_filter: str = "",
) -> Dict[str, Any]:
    attrs = job.get("attributes", {}) or {}

    company_id = (
        job.get("relationships", {})
        .get("company", {})
        .get("data", {})
        .get("id")
    )

    company = companies.get(company_id, {})

    salary_data = attrs.get("salary_data") or {}
    onet_data = attrs.get("onet_data") or {}
    recruiter_data = attrs.get("recruiter_data") or {}

    row = {
        "List": "",
        "source": source,
        "source_company_domain": source_company_domain,
        "global_location": global_location,
        "global_onet_codes": global_onet_codes,
        "company_location_filter": company_location_filter,
        "id": job.get("id"),
        "type": job.get("type"),
        "title": attrs.get("title"),
        "translated_title": attrs.get("translated_title"),
        "normalized_title": attrs.get("normalized_title"),
        "description": attrs.get("description"),
        "url": attrs.get("url"),
        "first_seen_at": attrs.get("first_seen_at"),
        "last_seen_at": attrs.get("last_seen_at"),
        "last_processed_at": attrs.get("last_processed_at"),
        "posted_at": attrs.get("posted_at"),
        "contract_types": safe_json(attrs.get("contract_types")),
        "categories": safe_json(attrs.get("categories")),
        "onet_code": onet_data.get("code"),
        "onet_family": onet_data.get("family"),
        "onet_occupation_name": onet_data.get("occupation_name"),
        "recruiter_name": recruiter_data.get("name"),
        "recruiter_title": recruiter_data.get("title"),
        "recruiter_contact": recruiter_data.get("contact"),
        "salary": attrs.get("salary"),
        "salary_low": salary_data.get("salary_low"),
        "salary_high": salary_data.get("salary_high"),
        "salary_currency": salary_data.get("salary_currency"),
        "salary_low_usd": salary_data.get("salary_low_usd"),
        "salary_high_usd": salary_data.get("salary_high_usd"),
        "salary_time_unit": salary_data.get("salary_time_unit"),
        "seniority": attrs.get("seniority"),
        "status": attrs.get("status"),
        "language": attrs.get("language"),
        "location": attrs.get("location"),
        "location_data": safe_json(attrs.get("location_data")),
        "tags": safe_json(attrs.get("tags")),
        "company_id": company.get("company_id") or company_id,
        "company_name": company.get("company_name"),
        "company_domain": company.get("company_domain"),
        "company_ticker": company.get("company_ticker"),
        "raw_json": safe_json(job),
    }

    return {field: row.get(field, "") for field in FIELDNAMES}


def fetch_company_jobs(cutoff: datetime) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    domains = read_company_domains()
    rows = []
    raw_payloads = []

    for index, domain in enumerate(domains, start=1):
        if len(rows) >= MAX_COMPANY_JOBS:
            print(f"[INFO] Reached MAX_COMPANY_JOBS={MAX_COMPANY_JOBS}")
            break

        print(f"\n[INFO] Company {index}/{len(domains)}: {domain}")

        for page in range(1, MAX_COMPANY_PAGES + 1):
            if len(rows) >= MAX_COMPANY_JOBS:
                break

            encoded_domain = quote(domain, safe="")
            url = f"{BASE_URL}/companies/{encoded_domain}/job_openings"

            try:
                payload = request_json(
                    url,
                    build_params(
                        page=page,
                        per_page=COMPANY_PER_PAGE,
                        location=COMPANY_LOCATION_FILTER,
                    ),
                )
            except Exception as exc:
                print(f"[ERROR] Failed for company {domain}, page {page}: {exc}")
                continue

            raw_payloads.append({
                "company_domain": domain,
                "page": page,
                "location": COMPANY_LOCATION_FILTER,
                "payload": payload,
            })

            companies = company_lookup(payload.get("included", []) or [])
            page_jobs = payload.get("data", []) or []

            if not page_jobs:
                break

            for job in page_jobs:
                if len(rows) >= MAX_COMPANY_JOBS:
                    break

                attrs = job.get("attributes", {}) or {}

                if not is_valid_predictleads_job(attrs, cutoff):
                    continue

                if not location_matches(attrs, COMPANY_LOCATION_FILTER):
                    continue

                rows.append(
                    flatten_job(
                        job=job,
                        companies=companies,
                        source="tracked company",
                        source_company_domain=domain,
                        company_location_filter=COMPANY_LOCATION_FILTER,
                    )
                )

            time.sleep(0.2)

    return rows, raw_payloads


def fetch_global_jobs(cutoff: datetime) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    raw_payloads = []

    for page in range(1, MAX_GLOBAL_PAGES + 1):
        if len(rows) >= MAX_GLOBAL_JOBS:
            print(f"[INFO] Reached MAX_GLOBAL_JOBS={MAX_GLOBAL_JOBS}")
            break

        print(f"\n[INFO] Global jobs page {page}/{MAX_GLOBAL_PAGES}")
        print(f"[INFO] Global location: {GLOBAL_LOCATION}")
        print(f"[INFO] Global O*NET codes: {GLOBAL_ONET_CODES}")

        url = f"{BASE_URL}/discover/job_openings"

        try:
            payload = request_json(
                url,
                build_params(
                    page=page,
                    per_page=GLOBAL_PER_PAGE,
                    location=GLOBAL_LOCATION,
                    onet_codes=GLOBAL_ONET_CODES,
                ),
            )
        except Exception as exc:
            print(f"[ERROR] Failed global page {page}: {exc}")
            continue

        raw_payloads.append({
            "page": page,
            "location": GLOBAL_LOCATION,
            "onet_codes": GLOBAL_ONET_CODES,
            "payload": payload,
        })

        companies = company_lookup(payload.get("included", []) or [])
        page_jobs = payload.get("data", []) or []

        if not page_jobs:
            break

        for job in page_jobs:
            if len(rows) >= MAX_GLOBAL_JOBS:
                break

            attrs = job.get("attributes", {}) or {}

            if not is_valid_predictleads_job(attrs, cutoff):
                continue

            if not location_matches(attrs, GLOBAL_LOCATION):
                continue

            rows.append(
                flatten_job(
                    job=job,
                    companies=companies,
                    source="global company",
                    global_location=GLOBAL_LOCATION,
                    global_onet_codes=GLOBAL_ONET_CODES,
                )
            )

        time.sleep(0.2)

    return rows, raw_payloads


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for row in rows:
        key = row.get("id") or row.get("url")

        if not key:
            key = f"{row.get('company_domain')}|{row.get('title')}|{row.get('location')}"

        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    return deduped


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


def main() -> None:
    if not API_KEY or not API_TOKEN:
        raise RuntimeError(
            "Missing PredictLeads credentials. Add both PREDICTLEADS_API_KEY and PREDICTLEADS_API_TOKEN as GitHub Secrets."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)
    ref = load_reference_data()

    print(f"[INFO] HOURS_BACK: {HOURS_BACK}")
    print(f"[INFO] Cutoff last_seen_at: {cutoff.isoformat()}")
    print(f"[INFO] GLOBAL_LOCATION: {GLOBAL_LOCATION}")
    print(f"[INFO] GLOBAL_ONET_CODES: {GLOBAL_ONET_CODES}")
    print(f"[INFO] COMPANY_LOCATION_FILTER: {COMPANY_LOCATION_FILTER}")
    print(f"[INFO] COMPANY_PER_PAGE: {COMPANY_PER_PAGE}")
    print(f"[INFO] GLOBAL_PER_PAGE: {GLOBAL_PER_PAGE}")
    print(f"[INFO] REQUEST_TIMEOUT_SECONDS: {REQUEST_TIMEOUT_SECONDS}")
    print(f"[INFO] MAX_COMPANIES: {MAX_COMPANIES}")
    print(f"[INFO] MAX_COMPANY_JOBS: {MAX_COMPANY_JOBS}")
    print(f"[INFO] MAX_GLOBAL_JOBS: {MAX_GLOBAL_JOBS}")
    print(f"[INFO] MAX_OUTPUT_ROWS: {MAX_OUTPUT_ROWS}")
    print(f"[INFO] FETCH_COMPANY_JOBS: {FETCH_COMPANY_JOBS}")
    print(f"[INFO] FETCH_GLOBAL_JOBS: {FETCH_GLOBAL_JOBS}")

    company_rows = []
    global_rows = []
    raw_company_payloads = []
    raw_global_payloads = []

    if FETCH_COMPANY_JOBS:
        company_rows, raw_company_payloads = fetch_company_jobs(cutoff)

    if FETCH_GLOBAL_JOBS:
        global_rows, raw_global_payloads = fetch_global_jobs(cutoff)

    company_rows = enrich_and_filter_rows(company_rows, ref)[:MAX_COMPANY_JOBS]
    global_rows = enrich_and_filter_rows(global_rows, ref)[:MAX_GLOBAL_JOBS]

    combined_rows = dedupe_rows(company_rows + global_rows)
    combined_rows = combined_rows[:MAX_OUTPUT_ROWS]

    write_csv(COMPANY_CSV_PATH, company_rows)
    write_csv(GLOBAL_CSV_PATH, global_rows)
    write_csv(COMBINED_CSV_PATH, combined_rows)

    write_json(RAW_COMPANY_JSON_PATH, raw_company_payloads)
    write_json(RAW_GLOBAL_JSON_PATH, raw_global_payloads)

    print("\n[DONE]")
    print(f"Company jobs saved: {len(company_rows)} -> {COMPANY_CSV_PATH}")
    print(f"Global jobs saved: {len(global_rows)} -> {GLOBAL_CSV_PATH}")
    print(f"Combined deduped jobs saved: {len(combined_rows)} -> {COMBINED_CSV_PATH}")
    print(f"Raw company JSON -> {RAW_COMPANY_JSON_PATH}")
    print(f"Raw global JSON -> {RAW_GLOBAL_JSON_PATH}")


if __name__ == "__main__":
    main()
