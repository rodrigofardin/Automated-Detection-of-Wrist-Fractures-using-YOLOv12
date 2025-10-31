"""
split_check.py

Módulo para dividir datasets de imagens médicas sem vazamento (patient-level split),
verificar potenciais vazamentos (paciente, estudo, arquivo, temporal, duplicatas)
e preparar estrutura YOLO (copiar imagens/labels) e atualizar data.yaml.

Objetivo: código limpo, testável e pronto para publicação.
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
# NOTE: do NOT call basicConfig at import time in a library - it may add handlers when the module
# is imported multiple times (or under different names) and cause duplicated log records.
def configure_logging(level: int = logging.INFO, fmt: str = "%(levelname)s: %(message)s") -> None:
    """Configure logging for CLI or notebooks. Call this once in the entrypoint (not on import)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=fmt)
    # also set module logger level explicitly
    logger.setLevel(level)


# ----------------- Core helpers -----------------

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Carrega dataset CSV e faz pequenas validações.

    Retorna DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Dataset carregado: %d amostras, %d colunas", len(df), df.shape[1])
    return df


def split_by_patient(df: pd.DataFrame, patient_col: str = "patient_id", seed: int = 42,
                     test_size: float = 0.30, val_frac_of_temp: float = 1/3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Faz split por paciente garantindo que pacientes não se sobreponham entre splits.

    - test_size: fração do total de pacientes reservada como temp (val+test)
    - val_frac_of_temp: quando aplicado em temp, fração que vira val (restante → test)
    """
    if patient_col not in df.columns:
        raise KeyError(f"Coluna de paciente '{patient_col}' não encontrada no DataFrame")

    patients = df[patient_col].astype(str).unique()
    train_p, temp_p = train_test_split(patients, test_size=test_size, random_state=seed)
    valid_p, test_p = train_test_split(temp_p, test_size=val_frac_of_temp, random_state=seed)

    train_df = df[df[patient_col].astype(str).isin(train_p)].reset_index(drop=True)
    val_df = df[df[patient_col].astype(str).isin(valid_p)].reset_index(drop=True)
    test_df = df[df[patient_col].astype(str).isin(test_p)].reset_index(drop=True)

    logger.info("Split feito: train=%d, val=%d, test=%d samples", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                out_dir: str | Path = "./splits", patient_col: str = "patient_id") -> None:
    """Salva CSVs dos splits e arquivos de pacientes por split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    pd.Series(train_df[patient_col].astype(str).unique()).to_csv(out_dir / "train_patients.txt", index=False, header=False)
    pd.Series(val_df[patient_col].astype(str).unique()).to_csv(out_dir / "val_patients.txt", index=False, header=False)
    pd.Series(test_df[patient_col].astype(str).unique()).to_csv(out_dir / "test_patients.txt", index=False, header=False)

    logger.info("Splits salvos em %s", out_dir)


# ----------------- Checks -----------------

def check_disjoint(a: Iterable, b: Iterable, name: str) -> None:
    """Verifica interseção entre duas coleções e lança AssertionError se houver overlap.

    Essa função usa assert para falhar rapidamente em pipelines de CI quando leak é detectado.
    """
    set_a, set_b = set(a), set(b)
    inter = set_a & set_b
    logger.info("%s: %d interseções", name, len(inter))
    if len(inter) > 0:
        raise AssertionError(f"Vazamento detectado em {name}: exemplo={list(inter)[:5]}")


def check_group_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, keys: List[str]) -> None:
    """Imprime contagens de interseção para chaves simples (patient_id, filestem, etc.)."""
    for key in keys:
        if key not in train_df.columns:
            logger.debug("Coluna %s não existe, pulando", key)
            continue
        s_tr, s_va, s_te = set(train_df[key].astype(str)), set(val_df[key].astype(str)), set(test_df[key].astype(str))
        logger.info("[%s] train∩val=%d, train∩test=%d, val∩test=%d", key, len(s_tr & s_va), len(s_tr & s_te), len(s_va & s_te))


def check_exact_row_duplicates(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, ignore_cols: Optional[List[str]] = None) -> None:
    """Checa duplicatas exatas de linhas entre splits ignorando colunas especificadas."""
    ignore_cols = ignore_cols or []

    def row_hash(d: pd.DataFrame) -> pd.Series:
        tmp = d.drop(columns=ignore_cols, errors='ignore').copy()
        tmp = tmp.reindex(sorted(tmp.columns), axis=1)
        return pd.util.hash_pandas_object(tmp, index=False).astype('uint64')

    h_tr, h_va, h_te = set(row_hash(train_df)), set(row_hash(val_df)), set(row_hash(test_df))
    logger.info("[duplicatas exatas] train∩val=%d, train∩test=%d, val∩test=%d", len(h_tr & h_va), len(h_tr & h_te), len(h_va & h_te))


def to_datetime_best_effort(s: pd.Series) -> pd.Series:
    """Tenta converter uma série para datetime tentando unidades numéricas comuns.

    Retorna pd.Series datetime (com NaT onde falhar).
    """
    if s.dtype == 'O':
        return pd.to_datetime(s, errors='coerce')
    if pd.api.types.is_numeric_dtype(s):
        for unit in ['s', 'ms', 'us', 'ns']:
            t = pd.to_datetime(s, errors='coerce', unit=unit)
            if t.notna().sum() > len(s) * 0.5:
                return t
    return pd.to_datetime(s, errors='coerce')


def check_temporal_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, datetime_col: str) -> None:
    """Checa se existe sobreposição temporal potencial entre splits (apenas informe, não assert)."""
    if datetime_col not in train_df.columns:
        logger.debug("Coluna datetime %s não existe", datetime_col)
        return
    tr = to_datetime_best_effort(train_df[datetime_col])
    va = to_datetime_best_effort(val_df[datetime_col])
    te = to_datetime_best_effort(test_df[datetime_col])
    logger.info("[temporal %s] max(train)=%s | min(val)=%s | min(test)=%s", datetime_col, tr.max(), va.min(), te.min())
    if (pd.notna(tr.max()) and pd.notna(va.min()) and tr.max() >= va.min()) or (pd.notna(tr.max()) and pd.notna(te.min()) and tr.max() >= te.min()):
        logger.warning("⚠️ Possível overlap temporal (ok se split por paciente)")


# ----------------- YOLO organization -----------------

def find_image_path(src_images: str | Path, filestem: str, exts: Optional[List[str]] = None) -> Optional[Path]:
    exts = exts or [".jpg", ".jpeg", ".png"]
    src_images = Path(src_images)
    for ext in exts:
        p = src_images / f"{filestem}{ext}"
        if p.exists():
            return p
    return None


def organize_yolo(splits_dir: str | Path, src_images: str | Path, src_labels: str | Path, dst_root: str | Path,
                  image_exts: Optional[List[str]] = None) -> None:
    """Copia imagens e labels para estrutura YOLO: dst_root/{train,val,test}/(images|labels).

    Exige que os arquivos train.csv, val.csv e test.csv no `splits_dir` possuam a coluna 'filestem'.
    """
    splits_dir = Path(splits_dir)
    dst_root = Path(dst_root)
    image_exts = image_exts or [".jpg", ".jpeg", ".png"]

    for split in ["train", "val", "test"]:
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        df = pd.read_csv(splits_dir / f"{split}.csv")
        if "filestem" not in df.columns:
            raise KeyError(f"Arquivo {split}.csv não contém a coluna 'filestem'")
        logger.info("📂 Movendo arquivos do split %s (%d amostras)", split, len(df))
        for _, row in df.iterrows():
            filestem = str(row["filestem"])
            img_path = find_image_path(src_images, filestem, exts=image_exts)
            lbl_path = Path(src_labels) / f"{filestem}.txt"
            if img_path is None:
                logger.warning("Imagem não encontrada para %s", filestem)
                continue
            dst_img = dst_root / split / "images" / img_path.name
            shutil.copy2(img_path, dst_img)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_root / split / "labels" / lbl_path.name)
            else:
                logger.warning("Label não encontrada para %s", filestem)

    logger.info("✅ Organização YOLO concluída em %s", dst_root)

    # --- try to copy and adapt a meta.yaml (if dataset provides one) ---
    try:
        splits_dir = Path(splits_dir)
        # look for a meta.yaml in the dataset folder structure (common pattern)
        candidates = list(splits_dir.parent.rglob('meta.yaml'))
        if candidates:
            src_meta = candidates[0]
            dst_meta_copy = Path(dst_root) / 'original_meta.yaml'
            shutil.copy2(src_meta, dst_meta_copy)
            logger.info('Copied original meta.yaml from %s to %s', src_meta, dst_meta_copy)

            # load and update paths
            with open(src_meta, 'r', encoding='utf-8') as f:
                meta = yaml.safe_load(f) or {}

            # remove placeholder path if it's present and looks like a FILL-IN
            if isinstance(meta.get('path'), str) and 'FILL' in meta.get('path'):
                meta.pop('path', None)

            # set train/val/test to the new image folders (absolute paths)
            meta['train'] = str((Path(dst_root) / 'train' / 'images').resolve())
            meta['val'] = str((Path(dst_root) / 'val' / 'images').resolve())
            meta['test'] = str((Path(dst_root) / 'test' / 'images').resolve())
            # update path entry to dst_root if desired (optional) - keep consistent
            meta['path'] = str(Path(dst_root).resolve())
            # remover a chave 'path' (não guardar caminho absoluto no meta)
            meta.pop('path', None)

            # write updated YAML into dst_root/data.yaml
            out_yaml = Path(dst_root) / 'data.yaml'
            with open(out_yaml, 'w', encoding='utf-8') as f:
                yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
            logger.info('Updated data.yaml written to %s', out_yaml)
        else:
            logger.debug('No meta.yaml found under %s; skipping meta copy/update', splits_dir.parent)
    except Exception as e:
        logger.warning('Failed to copy/update meta.yaml: %s', e)


# ----------------- CLI / runner -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset por paciente e verificar leaks")
    parser.add_argument("--csv", type=str, default="14825193/dataset.csv", help="Caminho para dataset CSV")
    parser.add_argument("--out", type=str, default="splits", help="Diretório de saída para os splits")
    parser.add_argument("--patient_col", type=str, default="patient_id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.csv)
    train_df, val_df, test_df = split_by_patient(df, patient_col=args.patient_col, seed=args.seed)
    save_splits(train_df, val_df, test_df, out_dir=args.out, patient_col=args.patient_col)

    # Checks (exemplos)
    keys_to_check = [k for k in [args.patient_col, 'study_number', 'filestem', 'filepath'] if k in df.columns]
    check_group_overlap(train_df, val_df, test_df, keys_to_check)
    if 'study_number' in df.columns:
        make_study_key = lambda d: (d[args.patient_col].astype(str) + "#" + d['study_number'].astype(str))
        check_disjoint(make_study_key(train_df), make_study_key(val_df), 'study_key train∩val')
    ignore_cols = [c for c in ['patient_id', 'study_number', 'timehash', 'filepath'] if c in df.columns]
    check_exact_row_duplicates(train_df, val_df, test_df, ignore_cols=ignore_cols)

    # temporal checks
    candidate_dates = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp'])]
    for c in candidate_dates[:3]:
        check_temporal_leakage(train_df, val_df, test_df, datetime_col=c)

    logger.info("Script finalizado")
