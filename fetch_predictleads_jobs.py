import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


BASE_URL = "https://predictleads.com/api/v3"

API_KEY = os.getenv("PREDICTLEADS_API_KEY", "").strip()
API_TOKEN = os.getenv("PREDICTLEADS_API_TOKEN", "").strip()

DAYS_BACK = int(os.getenv("DAYS_BACK", "7"))
PER_PAGE = int(os.getenv("PER_PAGE", "100"))
MAX_GLOBAL_PAGES = int(os.getenv("MAX_GLOBAL_PAGES", "1"))
MAX_COMPANY_PAGES = int(os.getenv("MAX_COMPANY_PAGES", "1"))

FETCH_COMPANY_JOBS = os.getenv("FETCH_COMPANY_JOBS", "true").lower() == "true"
FETCH_GLOBAL_JOBS = os.getenv("FETCH_GLOBAL_JOBS", "true").lower() == "true"

COMPANIES_CSV = Path("companies.csv")
OUTPUT_DIR = Path("output")

COMPANY_CSV_PATH = OUTPUT_DIR / "company_jobs.csv"
GLOBAL_CSV_PATH = OUTPUT_DIR / "global_jobs.csv"
COMBINED_CSV_PATH = OUTPUT_DIR / "combined_predictleads_jobs.csv"

RAW_COMPANY_JSON_PATH = OUTPUT_DIR / "raw_company_jobs.json"
RAW_GLOBAL_JSON_PATH = OUTPUT_DIR / "raw_global_jobs.json"


FIELDNAMES = [
    "source",
    "source_company_domain",

    "id",
    "type",

    "title",
    "translated_title",
    "normalized_title",
    "description",
    "url",

    "first_seen_at",
    "last_seen_at",
    "last_processed_at",
    "posted_at",

    "contract_types",
    "categories",

    "onet_code",
    "onet_family",
    "onet_occupation_name",

    "recruiter_name",
    "recruiter_title",
    "recruiter_contact",

    "salary",
    "salary_low",
    "salary_high",
    "salary_currency",
    "salary_low_usd",
    "salary_high_usd",
    "salary_time_unit",

    "seniority",
    "status",
    "language",

    "location",
    "location_data",
    "tags",

    "company_id",
    "company_name",
    "company_domain",
    "company_ticker",

    "raw_json",
]


def safe_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def is_recent_english_job(attrs: Dict[str, Any], cutoff: datetime) -> bool:
    if attrs.get("language") != "en":
        return False

    last_seen_at = parse_dt(attrs.get("last_seen_at"))
    if not last_seen_at:
        return False

    return last_seen_at >= cutoff


def build_params(page: int) -> Dict[str, Any]:
    return {
        "api_key": API_KEY,
        "api_token": API_TOKEN,
        "page": page,
        "per_page": PER_PAGE,
    }


def request_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    clean_params = {k: v for k, v in params.items() if v not in ["", None]}

    response = requests.get(
        url,
        params=clean_params,
        headers={"Accept": "application/json"},
        timeout=60,
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

        if "company_domain" not in reader.fieldnames:
            raise ValueError("companies.csv must have a header named company_domain")

        for row in reader:
            domain = (row.get("company_domain") or "").strip()
            domain = domain.replace("https://", "").replace("http://", "")
            domain = domain.replace("www.", "")
            domain = domain.split("/")[0].strip()

            if domain:
                domains.append(domain)

    unique_domains = sorted(set(domains))
    print(f"[INFO] Loaded company domains: {len(unique_domains)}")
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
        "source": source,
        "source_company_domain": source_company_domain,

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
        print(f"\n[INFO] Company {index}/{len(domains)}: {domain}")

        for page in range(1, MAX_COMPANY_PAGES + 1):
            encoded_domain = quote(domain, safe="")
            url = f"{BASE_URL}/companies/{encoded_domain}/job_openings"

            try:
                payload = request_json(url, build_params(page))
            except Exception as exc:
                print(f"[ERROR] Failed for company {domain}, page {page}: {exc}")
                continue

            raw_payloads.append({
                "company_domain": domain,
                "page": page,
                "payload": payload,
            })

            companies = company_lookup(payload.get("included", []) or [])

            page_jobs = payload.get("data", []) or []
            if not page_jobs:
                break

            for job in page_jobs:
                attrs = job.get("attributes", {}) or {}

                if not is_recent_english_job(attrs, cutoff):
                    continue

                rows.append(
                    flatten_job(
                        job=job,
                        companies=companies,
                        source="company",
                        source_company_domain=domain,
                    )
                )

            time.sleep(0.2)

    return rows, raw_payloads


def fetch_global_jobs(cutoff: datetime) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    raw_payloads = []

    for page in range(1, MAX_GLOBAL_PAGES + 1):
        print(f"\n[INFO] Global jobs page {page}/{MAX_GLOBAL_PAGES}")

        url = f"{BASE_URL}/discover/job_openings"

        try:
            payload = request_json(url, build_params(page))
        except Exception as exc:
            print(f"[ERROR] Failed global page {page}: {exc}")
            continue

        raw_payloads.append({
            "page": page,
            "payload": payload,
        })

        companies = company_lookup(payload.get("included", []) or [])

        page_jobs = payload.get("data", []) or []
        if not page_jobs:
            break

        for job in page_jobs:
            attrs = job.get("attributes", {}) or {}

            if not is_recent_english_job(attrs, cutoff):
                continue

            rows.append(
                flatten_job(
                    job=job,
                    companies=companies,
                    source="global",
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

    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

    print(f"[INFO] DAYS_BACK: {DAYS_BACK}")
    print(f"[INFO] Cutoff last_seen_at: {cutoff.isoformat()}")
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

    combined_rows = dedupe_rows(company_rows + global_rows)

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
