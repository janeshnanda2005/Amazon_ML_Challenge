"""
convert_to_parquet.py - Streaming TSV to Parquet Preprocessor
============================================================

Converts raw TSV files into compressed Parquet tables with precomputed
normalized blocking keys:
  - norm_name: lowercase, punctuation stripped, legal abbreviations expanded
  - norm_addr: lowercase, punctuation stripped, street abbreviations expanded
  - first_name_token: first core name token (excluding common generic tokens)
  - name_soundex: phonetic soundex of first core name token
  - postal_code: extracted 4-6 digit postal code
  - country_clean: lowercase cleaned country label
  - first_addr_token: first non-generic address token

Memory safety:
  Processes records in streaming batches (default 50,000 rows) using PyArrow.
  Never loads entire 20M-row datasets into pandas RAM.
  Peak RAM usage is kept strictly below 500 MB.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory for imports
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.normalize import (
    normalize_name,
    normalize_address,
    core_name_tokens,
    extract_postal_token,
    tokenize,
)
from src.similarity import soundex

PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("business_name", pa.string()),
    ("business_address", pa.string()),
    ("country", pa.string()),
    ("norm_name", pa.string()),
    ("norm_addr", pa.string()),
    ("first_name_token", pa.string()),
    ("name_soundex", pa.string()),
    ("postal_code", pa.string()),
    ("country_clean", pa.string()),
    ("first_addr_token", pa.string()),
])

GROUND_TRUTH_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()),
    ("matched_entity_ids", pa.string()),
])

GENERIC_ADDRESS_TOKENS = {
    "street", "road", "avenue", "lane", "drive", "boulevard", "highway",
    "near", "number", "floor", "building", "apartment", "the", "and", "of",
}


def _get_ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    ram = _get_ram_gb()
    print(f"[{ts}] [RAM: {ram:.2f} GB] {msg}", flush=True)


def convert_source_files(
    tsv_paths: list,
    parquet_path: Path,
    batch_size: int = 50000,
    force: bool = False,
    max_rows: int = None,
):
    """
    Stream one or more TSV source files and write to a single Parquet file
    with precomputed normalization columns.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists() and not force:
        try:
            meta = pq.read_metadata(parquet_path)
            if meta.num_rows > 0:
                _log(f"Parquet file already exists with {meta.num_rows:,} rows: {parquet_path} (skipping, pass --force to rebuild)")
                return
        except Exception:
            pass

    _log(f"Converting {[p.name for p in tsv_paths]} -> {parquet_path.name}...")
    t0 = time.time()
    total_rows = 0

    writer = pq.ParquetWriter(
        parquet_path,
        schema=PARQUET_SCHEMA,
        compression="snappy",
        use_dictionary=True,
    )

    con = duckdb.connect()
    # Configure DuckDB for safe memory bounds
    con.execute(f"PRAGMA max_memory = '{config.DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"PRAGMA threads = {config.LIGHTGBM_THREADS};")

    try:
        for tsv_path in tsv_paths:
            if not tsv_path.exists():
                raise FileNotFoundError(f"Source TSV not found: {tsv_path}")

            if max_rows is not None and total_rows >= max_rows:
                break

            _log(f"  Streaming: {tsv_path.name}...")
            # Use DuckDB CSV reader to stream in batches
            escaped_path = str(tsv_path).replace("\\", "/")
            query = f"""
                SELECT 
                    COALESCE(entity_id, '') AS entity_id,
                    COALESCE(business_name, '') AS business_name,
                    COALESCE(business_address, '') AS business_address,
                    COALESCE(country, '') AS country
                FROM read_csv('{escaped_path}', delim='\t', header=true, all_varchar=true)
            """

            cursor = con.cursor()
            cursor.execute(query)

            file_rows = 0
            while True:
                fetch_n = batch_size
                if max_rows is not None:
                    rem = max_rows - total_rows
                    if rem <= 0:
                        break
                    fetch_n = min(batch_size, rem)

                rows = cursor.fetchmany(fetch_n)
                if not rows:
                    break

                n_rows = len(rows)
                file_rows += n_rows
                total_rows += n_rows

                # Extract and normalize fields row-by-row in memory-efficient lists
                e_ids = []
                b_names = []
                b_addrs = []
                countries = []
                norm_names = []
                norm_addrs = []
                first_name_tokens = []
                name_soundexes = []
                postal_codes = []
                country_cleans = []
                first_addr_tokens = []

                for eid, name, addr, cntry in rows:
                    e_ids.append(eid)
                    b_names.append(name)
                    b_addrs.append(addr)
                    countries.append(cntry)

                    # Normalize name
                    nn = normalize_name(name)
                    norm_names.append(nn)

                    # Name tokens & soundex
                    toks = core_name_tokens(name, config.GENERIC_NAME_TOKENS)
                    first_tok = toks[0] if toks else ""
                    first_name_tokens.append(first_tok)
                    name_soundexes.append(soundex(first_tok) if first_tok else "")

                    # Normalize address
                    na = normalize_address(addr)
                    norm_addrs.append(na)

                    # Postal token
                    postal_codes.append(extract_postal_token(addr))

                    # Country clean
                    country_cleans.append(str(cntry).strip().lower())

                    # First distinct address token
                    a_toks = tokenize(na)
                    fat = ""
                    for at in a_toks:
                        if at not in GENERIC_ADDRESS_TOKENS and not at.isdigit():
                            fat = at
                            break
                    first_addr_tokens.append(fat)

                batch_table = pa.Table.from_arrays(
                    [
                        pa.array(e_ids, type=pa.string()),
                        pa.array(b_names, type=pa.string()),
                        pa.array(b_addrs, type=pa.string()),
                        pa.array(countries, type=pa.string()),
                        pa.array(norm_names, type=pa.string()),
                        pa.array(norm_addrs, type=pa.string()),
                        pa.array(first_name_tokens, type=pa.string()),
                        pa.array(name_soundexes, type=pa.string()),
                        pa.array(postal_codes, type=pa.string()),
                        pa.array(country_cleans, type=pa.string()),
                        pa.array(first_addr_tokens, type=pa.string()),
                    ],
                    schema=PARQUET_SCHEMA,
                )

                writer.write_table(batch_table)

                elapsed = time.time() - t0
                speed = total_rows / elapsed if elapsed > 0 else 0
                if total_rows % (batch_size * 4) == 0:
                    _log(f"    Processed {total_rows:,} rows total ({speed:.0f} rows/s)...")

            _log(f"  Finished {tsv_path.name}: {file_rows:,} rows.")

    finally:
        writer.close()
        con.close()

    elapsed = time.time() - t0
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    _log(f"Wrote {parquet_path.name}: {total_rows:,} rows ({size_mb:.1f} MB) in {elapsed:.1f}s ({total_rows/elapsed:.0f} rows/s)")


def convert_ground_truth_file(
    tsv_path: Path,
    parquet_path: Path,
    force: bool = False,
):
    """Convert train_ground_truth.tsv directly via DuckDB COPY."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists() and not force:
        try:
            meta = pq.read_metadata(parquet_path)
            if meta.num_rows > 0:
                _log(f"Ground truth parquet already exists with {meta.num_rows:,} rows: {parquet_path}")
                return
        except Exception:
            pass

    if not tsv_path.exists():
        _log(f"Ground truth file not found at {tsv_path}, skipping.")
        return

    _log(f"Converting ground truth: {tsv_path.name} -> {parquet_path.name}...")
    t0 = time.time()

    con = duckdb.connect()
    escaped_in = str(tsv_path).replace("\\", "/")
    escaped_out = str(parquet_path).replace("\\", "/")

    query = f"""
        COPY (
            SELECT 
                COALESCE(source1_entity_id, '') AS source1_entity_id,
                COALESCE(matched_entity_ids, '') AS matched_entity_ids
            FROM read_csv('{escaped_in}', delim='\t', header=true, all_varchar=true)
        ) TO '{escaped_out}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """
    con.execute(query)
    con.close()

    elapsed = time.time() - t0
    meta = pq.read_metadata(parquet_path)
    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    _log(f"Wrote {parquet_path.name}: {meta.num_rows:,} rows ({size_mb:.1f} MB) in {elapsed:.1f}s")


def convert_all(force: bool = False, max_rows: int = None, train_only: bool = False):
    """Convert all train (and optionally test) datasets into data/parquet/."""
    _log(f"Starting Parquet conversion (max_rows={max_rows}, train_only={train_only})...")

    # 1. Train Source 1
    convert_source_files(
        [config.TRAIN_SOURCE1],
        config.PARQUET_TRAIN_SOURCE1,
        batch_size=config.FEATURE_BATCH_SIZE * 5,
        force=force,
        max_rows=max_rows,
    )

    # 2. Train Candidate Pool (Source 2 + Source 3 unified)
    convert_source_files(
        [config.TRAIN_SOURCE2, config.TRAIN_SOURCE3],
        config.PARQUET_TRAIN_POOL,
        batch_size=config.FEATURE_BATCH_SIZE * 5,
        force=force,
        max_rows=max_rows * 4 if max_rows else None,
    )

    # 3. Train Ground Truth
    convert_ground_truth_file(
        config.TRAIN_GROUND_TRUTH,
        config.PARQUET_TRAIN_GROUND_TRUTH,
        force=force,
    )

    if not train_only:
        # 4. Test Source 1
        convert_source_files(
            [config.TEST_SOURCE1],
            config.PARQUET_TEST_SOURCE1,
            batch_size=config.FEATURE_BATCH_SIZE * 5,
            force=force,
            max_rows=max_rows,
        )

        # 5. Test Candidate Pool (Source 2 + Source 3 unified)
        convert_source_files(
            [config.TEST_SOURCE2, config.TEST_SOURCE3],
            config.PARQUET_TEST_POOL,
            batch_size=config.FEATURE_BATCH_SIZE * 5,
            force=force,
            max_rows=max_rows * 4 if max_rows else None,
        )

    _log("Parquet conversion completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TSV files to Parquet with precomputed normalized fields")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if Parquet files already exist")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows per file for quick testing")
    parser.add_argument("--train-only", action="store_true", help="Only convert training data")
    args = parser.parse_args()
    convert_all(force=args.force, max_rows=args.max_rows, train_only=args.train_only)

