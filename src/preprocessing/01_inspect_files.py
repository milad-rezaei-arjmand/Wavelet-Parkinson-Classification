from pathlib import Path

import numpy as np
import pandas as pd
from matio import load_from_mat


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FILE_PATH = DATA_DIR / "001.mat"



def describe_object(name: str, obj: object) -> None:
    """نمایش نوع و ساختار متغیر خوانده‌شده از فایل MAT."""

    print("=" * 70)
    print(f"Variable: {name}")
    print(f"Python type: {type(obj)}")

    if isinstance(obj, pd.DataFrame):
        print(f"Shape: {obj.shape}")
        print(f"Columns: {list(obj.columns)}")
        print("\nFirst rows:")
        print(obj.head())

        numeric_data = obj.select_dtypes(include=[np.number])

        if not numeric_data.empty:
            print(f"\nNumeric shape: {numeric_data.shape}")
            print(f"NaN count: {numeric_data.isna().sum().sum()}")
            print(
                "Inf count:",
                np.isinf(numeric_data.to_numpy(dtype=float)).sum(),
            )

    elif isinstance(obj, np.ndarray):
        print(f"Shape: {obj.shape}")
        print(f"Dtype: {obj.dtype}")

    else:
        print(f"Value preview: {repr(obj)[:500]}")


def main() -> None:
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    mat_data = load_from_mat(
        FILE_PATH,
        raw_data=False,
        add_table_attrs=True,
    )

    print(f"\nFile: {FILE_PATH.name}")
    print(f"Top-level variables: {list(mat_data.keys())}\n")

    for variable_name, variable_value in mat_data.items():
        if str(variable_name).startswith("__"):
            continue

        describe_object(
            name=str(variable_name),
            obj=variable_value,
        )


if __name__ == "__main__":
    main()


print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)
print("File path:", FILE_PATH)
print("File exists:", FILE_PATH.exists())