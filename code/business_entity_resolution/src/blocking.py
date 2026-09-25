"""
Candidate generation (blocking).

Provides:
  1. DuckDBBlocker: High-performance, disk-backed, multi-rule SQL blocking
     using DuckDB directly over Parquet files. Handles millions of records
     with strict memory bounds and prioritize-ranked candidate selection.
  2. BlockIndex: Fallback in-memory inverted index for small synthetic tests.
  3. Diagnostics: candidate_stats and blocking_recall for quantitative auditing.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd

from . import config
from .normalize import (
    core_name_tokens,
    normalize_address,
    normalize_name,
    tokenize,
    extract_postal_token,
)
from .similarity import soundex

# Address tokens too generic to block on by themselves.
GENERIC_ADDRESS_TOKENS = {
    "street", "road", "avenue", "lane", "drive", "boulevard", "highway",
    "near", "number", "floor", "building", "apartment", "the", "and", "of",
}


class DuckDBBlocker:
    """
    Disk-backed SQL blocking engine powered by DuckDB over Parquet tables.
    
    Supports 5 unioned blocking strategies:
      Rule 1: Exact normalized name match
      Rule 2: First core name token + postal code match
      Rule 3: Soundex phonetic key + postal code match
      Rule 4: Distinctive first name token + country match
      Rule 5: Distinctive address token + postal code match
      
    Candidates are deduplicated across rules. If a Source-1 entity exceeds
    MAX_CANDIDATES_PER_ENTITY, candidates are prioritized by rule confidence
    (min rule_id) rather than arbitrary truncation.
    """

    def __init__(
        self,
        pool_parquet_path: Union[str, Path],
        db_path: str = ":memory:",
        memory_limit: str = None,
        threads: int = None,
    ):
        self.pool_path = Path(pool_parquet_path)
        self.db_path = str(db_path)
        self.memory_limit = memory_limit or config.DUCKDB_MEMORY_LIMIT
        self.threads = threads or config.LIGHTGBM_THREADS
        
        self.con = duckdb.connect(self.db_path)
        self.con.execute(f"PRAGMA max_memory = '{self.memory_limit}';")
        self.con.execute(f"PRAGMA threads = {self.threads};")
        
        # Register the candidate pool Parquet view
        escaped_pool = str(self.pool_path).replace("\\", "/")
        self.con.execute(f"CREATE OR REPLACE VIEW pool AS SELECT * FROM read_parquet('{escaped_pool}');")

    def block_chunk(
        self,
        source1_chunk: pd.DataFrame,
        max_candidates: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate candidate pairs for a chunk of Source 1 records.
        
        Parameters
        ----------
        source1_chunk : DataFrame with Source 1 records (must have normalized
                        fields: norm_name, first_name_token, name_soundex,
                        postal_code, country_clean, first_addr_token).
        max_candidates : Maximum candidates to keep per Source-1 entity
                         (ranked by rule confidence).
        
        Returns
        -------
        DataFrame with columns [source1_entity_id, candidate_entity_id]
        """
        max_cand = max_candidates or config.MAX_CANDIDATES_PER_ENTITY
        
        # Ensure derived columns exist in s1_chunk
        s1 = source1_chunk
        if "norm_name" not in s1.columns:
            s1 = s1.copy()
            s1["norm_name"] = s1["business_name"].map(normalize_name)
            s1["norm_addr"] = s1["business_address"].map(normalize_address)
            core_toks = s1["business_name"].map(lambda n: core_name_tokens(n, config.GENERIC_NAME_TOKENS))
            s1["first_name_token"] = core_toks.map(lambda t: t[0] if t else "")
            s1["name_soundex"] = s1["first_name_token"].map(soundex)
            s1["postal_code"] = s1["business_address"].map(extract_postal_token)
            s1["country_clean"] = s1["country"].map(lambda c: str(c).strip().lower())
            
            def _fat(addr):
                for tok in tokenize(normalize_address(addr)):
                    if tok not in GENERIC_ADDRESS_TOKENS and not tok.isdigit():
                        return tok
                return ""
            s1["first_addr_token"] = s1["business_address"].map(_fat)

        # Register chunk into DuckDB
        self.con.register("s1_chunk", s1)

        query = f"""
            WITH cands AS (
                -- Rule 1: Exact normalized name match (high precision)
                SELECT s1.entity_id AS source1_entity_id, pool.entity_id AS candidate_entity_id, 1 AS rule_id
                FROM s1_chunk s1
                JOIN pool ON s1.norm_name = pool.norm_name AND length(s1.norm_name) > 2

                UNION ALL

                -- Rule 2: First name token + postal code match
                SELECT s1.entity_id AS source1_entity_id, pool.entity_id AS candidate_entity_id, 2 AS rule_id
                FROM s1_chunk s1
                JOIN pool ON s1.first_name_token = pool.first_name_token 
                         AND s1.postal_code = pool.postal_code 
                         AND s1.postal_code != '' 
                         AND length(s1.first_name_token) >= 2

                UNION ALL

                -- Rule 3: Soundex phonetic key + postal code match
                SELECT s1.entity_id AS source1_entity_id, pool.entity_id AS candidate_entity_id, 3 AS rule_id
                FROM s1_chunk s1
                JOIN pool ON s1.name_soundex = pool.name_soundex 
                         AND s1.postal_code = pool.postal_code 
                         AND s1.postal_code != ''

                UNION ALL

                -- Rule 4: First name token + country match (for distinctive tokens)
                SELECT s1.entity_id AS source1_entity_id, pool.entity_id AS candidate_entity_id, 4 AS rule_id
                FROM s1_chunk s1
                JOIN pool ON s1.first_name_token = pool.first_name_token 
                         AND s1.country_clean = pool.country_clean
                         AND length(s1.first_name_token) >= 4

                UNION ALL

                -- Rule 5: Address token + postal code match
                SELECT s1.entity_id AS source1_entity_id, pool.entity_id AS candidate_entity_id, 5 AS rule_id
                FROM s1_chunk s1
                JOIN pool ON s1.first_addr_token = pool.first_addr_token 
                         AND s1.postal_code = pool.postal_code 
                         AND s1.postal_code != '' 
                         AND length(s1.first_addr_token) >= 3
            ),
            deduped AS (
                SELECT 
                    source1_entity_id,
                    candidate_entity_id,
                    MIN(rule_id) AS min_rule_id
                FROM cands
                GROUP BY source1_entity_id, candidate_entity_id
            ),
            ranked AS (
                SELECT 
                    source1_entity_id,
                    candidate_entity_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY source1_entity_id 
                        ORDER BY min_rule_id ASC, candidate_entity_id ASC
                    ) AS rn
                FROM deduped
            )
            SELECT source1_entity_id, candidate_entity_id
            FROM ranked
            WHERE rn <= {max_cand}
        """

        pairs_df = self.con.execute(query).df()
        self.con.unregister("s1_chunk")
        return pairs_df

    def fetch_candidate_pool_records(self, candidate_ids: list) -> pd.DataFrame:
        """Fetch full attributes for a batch of candidate IDs from pool."""
        if not candidate_ids:
            return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
            
        cand_df = pd.DataFrame({"cand_id": candidate_ids})
        self.con.register("needed_ids", cand_df)
        query = """
            SELECT p.entity_id, p.business_name, p.business_address, p.country,
                   p.norm_name, p.norm_addr, p.postal_code, p.country_clean
            FROM pool p
            JOIN (SELECT DISTINCT cand_id FROM needed_ids) n ON p.entity_id = n.cand_id
        """
        res = self.con.execute(query).df()
        self.con.unregister("needed_ids")
        return res

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


class BlockIndex:
    """Inverted indexes over a pool of candidate records (Source 2 + 3) for small in-memory runs."""

    def __init__(self, records: pd.DataFrame):
        self.name_token_index = defaultdict(set)
        self.soundex_index = defaultdict(set)
        self.postal_index = defaultdict(set)
        self.addr_token_index = defaultdict(set)
        self._build(records)

    def _build(self, records: pd.DataFrame):
        for row in records.itertuples(index=False):
            entity_id = row.entity_id
            name_tokens = core_name_tokens(
                row.business_name, config.GENERIC_NAME_TOKENS
            )
            for tok in name_tokens:
                self.name_token_index[tok].add(entity_id)

            if name_tokens:
                key = soundex(name_tokens[0])
                self.soundex_index[key].add(entity_id)

            postal = extract_postal_token(row.business_address)
            if postal:
                self.postal_index[postal].add(entity_id)

            addr_tokens = tokenize(normalize_address(row.business_address))
            for tok in addr_tokens:
                if tok in GENERIC_ADDRESS_TOKENS or tok.isdigit():
                    continue
                self.addr_token_index[tok].add(entity_id)

    def candidates_for(self, business_name: str, business_address: str) -> set:
        candidates = set()
        name_tokens = core_name_tokens(business_name, config.GENERIC_NAME_TOKENS)
        for tok in name_tokens:
            candidates |= self.name_token_index.get(tok, set())

        if name_tokens:
            key = soundex(name_tokens[0])
            candidates |= self.soundex_index.get(key, set())

        postal = extract_postal_token(business_address)
        if postal:
            candidates |= self.postal_index.get(postal, set())

        addr_tokens = tokenize(normalize_address(business_address))
        for tok in addr_tokens:
            if tok in GENERIC_ADDRESS_TOKENS or tok.isdigit():
                continue
            candidates |= self.addr_token_index.get(tok, set())

        return candidates


def generate_candidate_pairs(
    source1: pd.DataFrame, index: BlockIndex, max_candidates: int = None
) -> pd.DataFrame:
    """Generate candidate pairs in memory using BlockIndex (used for synthetic/smoke tests)."""
    max_candidates = max_candidates or config.MAX_CANDIDATES_PER_ENTITY
    rows = []
    for row in source1.itertuples(index=False):
        candidates = index.candidates_for(row.business_name, row.business_address)
        if len(candidates) > max_candidates:
            candidates = set(sorted(candidates)[:max_candidates])
        for cand_id in candidates:
            rows.append((row.entity_id, cand_id))

    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def candidate_stats(candidate_pairs: pd.DataFrame, num_source1_entities: int) -> dict:
    """
    Compute distribution metrics over generated candidate sets.
    Reports: avg, median, p95, p99, max, total.
    """
    if len(candidate_pairs) == 0:
        return {
            "avg": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
            "total_pairs": 0,
        }
    counts = candidate_pairs.groupby("source1_entity_id").size()
    zeros = max(0, num_source1_entities - len(counts))
    if zeros > 0:
        all_counts = pd.concat([counts, pd.Series([0] * zeros)])
    else:
        all_counts = counts

    return {
        "avg": float(all_counts.mean()),
        "median": float(all_counts.median()),
        "p95": float(all_counts.quantile(0.95)),
        "p99": float(all_counts.quantile(0.99)),
        "max": int(all_counts.max()),
        "total_pairs": len(candidate_pairs),
    }


def blocking_recall(
    candidate_pairs: pd.DataFrame, ground_truth: pd.DataFrame
) -> dict:
    """
    Diagnostic: what fraction of true matches survive blocking?
    """
    candidate_set = set(
        zip(candidate_pairs["source1_entity_id"], candidate_pairs["candidate_entity_id"])
    )

    total_true_matches = 0
    recovered_true_matches = 0

    for row in ground_truth.itertuples(index=False):
        matched_ids = [
            m.strip() for m in str(row.matched_entity_ids).split(",") if m.strip()
        ]
        total_true_matches += len(matched_ids)
        for mid in matched_ids:
            if (row.source1_entity_id, mid) in candidate_set:
                recovered_true_matches += 1

    recall = (
        recovered_true_matches / total_true_matches if total_true_matches else 1.0
    )
    return {
        "true_matches_total": total_true_matches,
        "true_matches_recovered": recovered_true_matches,
        "blocking_recall": recall,
        "candidate_pairs_total": len(candidate_set),
    }
