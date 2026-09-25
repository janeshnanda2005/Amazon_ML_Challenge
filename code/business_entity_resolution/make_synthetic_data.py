"""
Generates a small synthetic train + test set with the exact schema described
in the challenge README, purely so the pipeline can be exercised end-to-end
in an environment without the real (and here, inaccessible/huge) dataset.
NOT part of the submitted pipeline - delete before packaging your real run.
"""
import random
import csv
from pathlib import Path

random.seed(0)

ROOT = Path(__file__).resolve().parent
NAME_BASES = [
    "Sunrise Bakery", "Kumar Electronics", "Blue Ridge Logistics",
    "Om Sai Traders", "Golden Gate Consulting", "Patel Textiles",
    "Liberty Hardware", "Shree Ganesh Motors", "Cascade Software Inc",
    "Rajesh General Store", "Northstar Freight", "Anand Pharmacy",
    "Evergreen Landscaping", "Sharma Book Depot", "Pioneer Auto Parts",
]
NAME_SUFFIXES = ["", " Corp", " Corporation", " Pvt Ltd", " LLC", " & Co", " Inc"]
NAME_TYPOS = {
    "Sunrise": ["Sunrize", "Sun Rise"],
    "Electronics": ["Electroniks", "Electronic"],
}
STREETS = ["Main St", "5th Ave", "MG Road", "Park Lane", "Church Rd", "Station Rd"]
CITIES_US = ["Springfield", "Fairview", "Riverside", "Georgetown"]
CITIES_IN = ["Pune", "Nagpur", "Indore", "Surat"]
COUNTRIES = ["US", "India"]


def jitter_name(name):
    for word, typos in NAME_TYPOS.items():
        if word in name and random.random() < 0.3:
            name = name.replace(word, random.choice(typos))
    return name


def make_address(country):
    street = random.choice(STREETS)
    num = random.randint(1, 999)
    city = random.choice(CITIES_US if country == "US" else CITIES_IN)
    pin = random.randint(10000, 99999) if random.random() > 0.15 else None
    addr = f"{num} {street}, {city}"
    if pin:
        addr += f" {pin}"
    return addr


def write_tsv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def build_split(n_entities, source1_prefix, countries, start_idx=1):
    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
    s2_id, s3_id = 1, 1

    for i in range(start_idx, start_idx + n_entities):
        s1_id = f"{source1_prefix}{i:05d}"
        base = random.choice(NAME_BASES)
        suffix = random.choice(NAME_SUFFIXES)
        country = random.choice(countries)
        name = base + suffix
        address = make_address(country)
        s1_rows.append([s1_id, name, address, country])

        matched_ids = []
        n_matches = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
        for _ in range(n_matches):
            target_source = random.choice(["S2", "S3"])
            noisy_name = jitter_name(base) + random.choice(NAME_SUFFIXES)
            noisy_addr = address  # keep address consistent -> strong signal
            if target_source == "S2":
                rec_id = f"S2-{s2_id:05d}"
                s2_id += 1
                s2_rows.append([rec_id, noisy_name, noisy_addr, country])
            else:
                rec_id = f"S3-{s3_id:05d}"
                s3_id += 1
                s3_rows.append([rec_id, noisy_name, noisy_addr, country])
            matched_ids.append(rec_id)

        gt_rows.append([s1_id, ",".join(matched_ids)])

    # Add pure noise (non-matching) records to source2/3 so blocking has to
    # actually discriminate, not just return everything.
    for _ in range(n_entities):
        country = random.choice(countries)
        rec = [f"S2-{s2_id:05d}", random.choice(NAME_BASES) + random.choice(NAME_SUFFIXES),
               make_address(country), country]
        s2_id += 1
        s2_rows.append(rec)
    for _ in range(n_entities):
        country = random.choice(countries)
        rec = [f"S3-{s3_id:05d}", random.choice(NAME_BASES) + random.choice(NAME_SUFFIXES),
               make_address(country), country]
        s3_id += 1
        s3_rows.append(rec)

    return s1_rows, s2_rows, s3_rows, gt_rows


HEADER_SRC = ["entity_id", "business_name", "business_address", "country"]
HEADER_GT = ["source1_entity_id", "matched_entity_ids"]

# Train split: US + India only (per the README)
train_s1, train_s2, train_s3, train_gt = build_split(300, "S1-", ["US", "India"])
write_tsv(ROOT / "dataset/train/train_source1.tsv", train_s1, HEADER_SRC)
write_tsv(ROOT / "dataset/train/train_source2.tsv", train_s2, HEADER_SRC)
write_tsv(ROOT / "dataset/train/train_source3.tsv", train_s3, HEADER_SRC)
write_tsv(ROOT / "dataset/train/train_ground_truth.tsv", train_gt, HEADER_GT)

# Test split: US + India + France (unseen country, per the README)
test_s1, test_s2, test_s3, _ = build_split(100, "S1-", ["US", "India", "France"], start_idx=1)
write_tsv(ROOT / "dataset/test/test_source1.tsv", test_s1, HEADER_SRC)
write_tsv(ROOT / "dataset/test/test_source2.tsv", test_s2, HEADER_SRC)
write_tsv(ROOT / "dataset/test/test_source3.tsv", test_s3, HEADER_SRC)

print("Synthetic dataset written.")
print(f"train: s1={len(train_s1)} s2={len(train_s2)} s3={len(train_s3)} gt={len(train_gt)}")
print(f"test:  s1={len(test_s1)} s2={len(test_s2)} s3={len(test_s3)}")
