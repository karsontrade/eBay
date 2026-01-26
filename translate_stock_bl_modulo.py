#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate_stock_bl_modulo.py — LostArk CSV → 4 Baselinker XMLs (Stable Split)

Vlastnosti:
1. Rozdeľuje produkty do 4 súborov STABILNE (podľa SKU).
2. Produkt sa vždy nachádza v rovnakom súbore.
3. Rieši problém s "preblikávaním" produktov pri aktualizácii.
"""

from __future__ import annotations
import re, sys, requests, pandas as pd
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

# ────────────────────────── KONŠTANTY ──────────────────────────
DEFAULT_URL   = "http://lostark.pl/stock/current_stock.csv"
BASE_FILENAME = "feed_part" 
NUM_PARTS     = 4  # Rozdelíme to na 4 súbory (bezpečne pod 30MB)
BASE          = Path(__file__).parent

# Súbory s prekladmi (EN)
NORMAL_CSV = BASE / "normal_tokens_en.csv"
SPECIAL_CSV = BASE / "special_tokens_en.csv"
PHRASE_CSV = BASE / "phrase_translations_en.csv"
CATEGORY_CSV = BASE / "category_segments_en.csv"
CARTYPE_CSV = BASE / "cartype_translations_en.csv"
CARMODEL_CSV = BASE / "carmodel_translations_en.csv"

PACKAGE_MARGIN_CSV = BASE / "package_margins.csv"
SHIPPING_CSV = BASE / "shipping_price.csv"
PRICE_MARGIN_CSV = BASE / "price_margins.csv"
WEIGHT_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# ────────────────────────── LOGIKA ROZDEĽOVANIA ────────────────
def get_file_index(sku):
    """Vráti index súboru (0-3) na základe SKU. Zaručuje stabilitu."""
    try:
        # Ak je SKU číslo, použijeme modulo
        val = int(sku)
    except ValueError:
        # Ak SKU obsahuje písmená, spočítame ASCII hodnoty
        val = sum(ord(c) for c in str(sku))
    
    return val % NUM_PARTS

# ────────────────────────── POMOCNÉ FUNKCIE ────────────────────
def clean_text_for_xml(text: str) -> str:
    if not text: return ""
    return text.replace("“", '"').replace("”", '"').replace("„", '"').strip()

# (Tu sú načítavacie funkcie rovnaké ako predtým - skrátené)
def load_shipping(path): 
    if not path.exists(): return {}
    df = pd.read_csv(path, sep=";", comment="#", dtype=str).fillna("")
    return {r["package_size"].strip().upper(): {"price": float(r["price"].replace(",", ".")), "method": r["delivery_id"].strip()} for _, r in df.iterrows()}
SHIPPING = load_shipping(SHIPPING_CSV)

def load_package_margins(path):
    if not path.exists(): return {}
    df = pd.read_csv(path, sep=";", comment="#", dtype=str).fillna("")
    return {r["package_size"].strip().upper(): float(r["margin"].replace(",", ".")) for _, r in df.iterrows()}
MARGIN = load_package_margins(PACKAGE_MARGIN_CSV)

def load_price_margins(path):
    if not path.exists(): return []
    df = pd.read_csv(path, sep=";", comment="#", dtype=str).fillna("")
    return [{"min": float(r["min_price"].replace(",", ".")), "max": float(r["max_price"].replace(",", ".")), "percent": float(r["margin_percent"].replace(",", "."))} for _, r in df.iterrows()]
PRICE_MARGINS = load_price_margins(PRICE_MARGIN_CSV)

def price_based_margin(price_eur):
    for band in PRICE_MARGINS:
        if band["min"] <= price_eur <= band["max"]: return price_eur * (band["percent"] / 100)
    return 0.0

PLN_EUR_RATE = 1 / 4.23 
def apply_margin(price_pln, size):
    try: base_pln = float(price_pln.replace(",", "."))
    except ValueError: return 0.0
    base_eur = base_pln * PLN_EUR_RATE
    return round(base_eur + price_based_margin(base_eur) + MARGIN.get(size.strip().upper(), 0.0), 2)

def download_csv(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content), sep=";", dtype=str, encoding="utf-8").fillna("")

# Preklady
def load_map(path, upper=True):
    if not path.exists(): return []
    df = pd.read_csv(path, sep=";", dtype=str).fillna("")
    rows = df.to_dict("records")
    if upper: 
        for r in rows: r["original"] = r["original"].upper()
    return rows

norm = load_map(NORMAL_CSV); normal = {r["original"]: r["translation"] for r in norm}
spec = load_map(SPECIAL_CSV); special = {r["original"]: r["translation"] for r in spec}
phrs = load_map(PHRASE_CSV); phrase = {r["original"].upper(): r["translation"] for r in phrs}
phrase_keys = sorted(phrase, key=lambda k: len(k.split()), reverse=True)
cats = load_map(CATEGORY_CSV, upper=False); cat_map = {r["original"]: r["translation"] for r in cats}
ctype = load_map(CARTYPE_CSV); ctype_map, ctype_keys_list = (lambda d: ({r["original"]:r["translation"] for r in d}, sorted({r["original"]:r["translation"] for r in d}, key=lambda k: len(k.split()), reverse=True)))(load_map(CARTYPE_CSV))
cmod = load_map(CARMODEL_CSV); cmod_map, cmod_keys_list = (lambda d: ({r["original"]:r["translation"] for r in d}, sorted({r["original"]:r["translation"] for r in d}, key=lambda k: len(k.split()), reverse=True)))(load_map(CARMODEL_CSV))

def translate_name(text):
    t = text.upper().strip()
    for ph in phrase_keys:
        if ph in t: t = re.sub(rf"\b{re.escape(ph)}\b", phrase[ph].upper(), t)
    output = []
    for tok in re.split(r"\s+", t):
        if tok in normal and normal[tok]: output.append(normal[tok])
        elif tok in special and special[tok]: output.append(special[tok])
        elif tok: output.append(tok)
    return " ".join(output)

def replace_from_map(txt, mp, keys):
    t = txt.upper()
    for k in keys: t = re.sub(rf"\b{re.escape(k)}\b", mp[k].upper(), t)
    return re.sub(r"\s+", " ", t).strip()

translate_category = lambda p: " | ".join(cat_map.get(seg.strip(), seg.strip()) for seg in p.split(">"))
translate_cartype = lambda t: replace_from_map(t, ctype_map, ctype_keys_list)
translate_carmodel = lambda t: replace_from_map(t, cmod_map, cmod_keys_list)

def has_numbers(word): return any(char.isdigit() for char in word)
def smart_case_text(text, is_part_name=False):
    if not text: return ""
    tokens = re.findall(r'\(.*?\)|[\w-]+|[^\w\s]', text)
    processed = []
    for t in tokens:
        if t.startswith('(') or has_numbers(t) or (len(t)<=3 and t.isalpha()): processed.append(t.upper())
        elif is_part_name: processed.append(t.lower())
        else: processed.append(t.capitalize())
    return " ".join(processed)

def process(df):
    df = df.copy()
    if "sn" in df.columns: df = df[df["sn"].str.strip().astype(bool)]
    if "category_path" in df.columns: df = df[~df["category_path"].str.startswith("Opony i felgi")]
    if "package_size" in df.columns: df = df[df["package_size"].str.strip().str.upper().isin(["A","B","C","D","E"])]

    df["price_eur"] = df.apply(lambda r: apply_margin(r["price"], r["package_size"]), axis=1)
    df["category_path"] = df["category_path"].map(translate_category)
    df["cartype"] = df["cartype"].map(translate_cartype)
    df["name_translated"] = df["name"].map(translate_name)
    df["carmodel_tr"] = df["carmodel"].map(translate_carmodel)
    
    def compose_name(row):
        n = smart_case_text(str(row["name_translated"]).strip(), True)
        if n: n = n[0].upper() + n[1:]
        b = smart_case_text(str(row["brand"]).strip(), False)
        m = smart_case_text(str(row["carmodel_tr"]).strip(), False)
        sn = str(row["sn"]).strip().upper()
        return clean_text_for_xml(" ".join([part for part in [n, b, m, sn] if part]))

    df["final_title"] = df.apply(compose_name, axis=1)
    
    # PRIDANIE ID SÚBORU (0, 1, 2, 3)
    df["file_part_id"] = df["item_number"].apply(get_file_index)
    
    return df

# ────────────────────────── XML GENERÁTOR ──────────────────────
def generate_xml_file(df_part: pd.DataFrame, filename: str):
    root = ET.Element("products")
    
    for _, r in df_part.iterrows():
        p = ET.SubElement(root, "product")
        ET.SubElement(p, "sku").text = str(r["item_number"])
        
        if "ean" in r and r["ean"] and str(r["ean"]).strip():
             ET.SubElement(p, "ean").text = clean_text_for_xml(str(r["ean"]))

        ET.SubElement(p, "name").text = clean_text_for_xml(str(r["final_title"]))
        ET.SubElement(p, "quantity").text = "1" 
        ET.SubElement(p, "category_name").text = clean_text_for_xml(str(r["category_path"]))
        ET.SubElement(p, "manufacturer_name").text = clean_text_for_xml(str(r["brand"]))
        ET.SubElement(p, "price").text = f"{r['price_eur']:.2f}"
        ET.SubElement(p, "tax_rate").text = "23%"
        
        if "purchase_price" in r and r["purchase_price"] and str(r["purchase_price"]).strip():
            ET.SubElement(p, "purchase_price").text = str(r["purchase_price"])

        weight = WEIGHT_MAP.get(str(r["package_size"]).strip().upper(), 0)
        ET.SubElement(p, "weight").text = str(weight)
        ET.SubElement(p, "width").text = "0"; ET.SubElement(p, "height").text = "0"; ET.SubElement(p, "length").text = "0"

        desc = [str(r["final_title"]), f"Spare part removed from {r['cartype']}.", "It may fit other models and engines as well."]
        if r["sn"]: desc.append(f"OEM: {r['sn']}.")
        ET.SubElement(p, "description").text = clean_text_for_xml(" ".join(desc))
        ET.SubElement(p, "description_extra_1").text = "Please verify the condition of the part from the photos."

        if r["main_photo"]: ET.SubElement(p, "image").text = r["main_photo"]
        idx = 1
        for i in range(1, 10):
            if f"photo{i}" in r and r[f"photo{i}"]:
                ET.SubElement(p, f"image_extra_{idx}").text = r[f"photo{i}"]
                idx += 1

    with open(filename, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        tree = ET.ElementTree(root)
        if sys.version_info >= (3, 9): ET.indent(tree, space="\t", level=0)
        tree.write(f, encoding="utf-8", xml_declaration=False)

def export_parts(df: pd.DataFrame):
    print(f"📦 Total items: {len(df)}. Splitting into {NUM_PARTS} fixed parts.")
    
    for i in range(NUM_PARTS):
        # Vyfiltrujeme len tie produkty, ktoré patria do súboru 'i'
        df_part = df[df["file_part_id"] == i]
        
        filename = f"{BASE_FILENAME}_{i+1}.xml"
        print(f"   Writing {filename} ({len(df_part)} items)...")
        generate_xml_file(df_part, filename)

# ────────────────────────── MAIN ───────────────────────────────
def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print("⇣ Downloading CSV...")
    df_raw = download_csv(url)
    
    print("⚙️ Processing...")
    df = process(df_raw)
    
    print(f"📉 Total items after filter: {len(df)}")
    export_parts(df)
    print("✓ All 4 parts generated successfully.")

if __name__ == "__main__":
    main()