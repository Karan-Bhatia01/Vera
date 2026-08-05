import sys

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException


MAX_DATASETS_PER_USER = 3
MAX_ROWS = 35_000
MAX_COLUMNS = 20



def handle_upload(file, owner_email: str):

    try:

        ingestion = DataIngestion()

        existing_count = ingestion.count_files_for_owner(owner_email)

        if existing_count >= MAX_DATASETS_PER_USER:
            raise ValueError(
                f"Dataset limit reached ({MAX_DATASETS_PER_USER} max). "
                "Delete an existing dataset before uploading a new one."
            )

        # Peek at the file to validate shape before committing it to GridFS.
        # file.stream is read once here, then we reset the pointer so
        # store_file() can read it again from the start.
        file_bytes = file.read()
        file.seek(0)

        try:
            df = DataIngestion._read_csv_robust(file_bytes)
        except Exception:
            raise ValueError("Could not parse the uploaded file as a valid CSV.")

        rows, cols = df.shape

        if rows > MAX_ROWS:
            raise ValueError(
                f"Dataset has {rows} rows, which exceeds the {MAX_ROWS}-row limit."
            )

        if cols > MAX_COLUMNS:
            raise ValueError(
                f"Dataset has {cols} columns, which exceeds the {MAX_COLUMNS}-column limit."
            )

        ingestion.store_file(file, owner_email=owner_email)

        return {
            "filename": file.filename,
            "rows": rows,
            "columns": cols,
        }

    except ValueError:
        raise

    except Exception as e:
        raise CustomException(e, sys) from e