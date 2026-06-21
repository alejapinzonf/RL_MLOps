from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


SUPPORTED_EXTENSIONS = [".csv", ".xlsx"]


def find_data_files(raw_data_dir: Path) -> list[Path]:
    """
    Busca archivos CSV y Excel dentro de data/raw.
    """
    files = []

    for extension in SUPPORTED_EXTENSIONS:
        files.extend(raw_data_dir.glob(f"*{extension}"))

    return sorted(files)


def read_file(file_path: Path) -> pd.DataFrame:
    """
    Lee un archivo CSV o Excel y retorna un DataFrame.
    """
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)

    elif file_path.suffix == ".xlsx":
        df = pd.read_excel(file_path)

    else:
        raise ValueError(f"Formato no soportado: {file_path.suffix}")

    df["source_file"] = file_path.name
    return df


def load_historical_data(raw_data_dir: Path) -> pd.DataFrame:
    """
    Carga todos los archivos históricos encontrados en data/raw.
    """
    files = find_data_files(raw_data_dir)

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV o Excel en: {raw_data_dir}"
        )

    dataframes = []

    print("Archivos encontrados:")

    for file_path in files:
        print(f"  - {file_path.name}")
        df = read_file(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def main():
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_historical_data(RAW_DATA_DIR)

    output_path = INTERIM_DATA_DIR / "historical_loaded.csv"
    df.to_csv(output_path, index=False)

    print("\nCarga completada.")
    print(f"Filas cargadas: {df.shape[0]}")
    print(f"Columnas cargadas: {df.shape[1]}")
    print(f"Archivo generado: {output_path}")

    print("\nPrimeras filas:")
    print(df.head())


if __name__ == "__main__":
    main()
