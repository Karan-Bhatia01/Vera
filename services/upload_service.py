import sys
import threading

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.llm import AnalysisExplainer
from src.logger import logging
from src.exception import CustomException


MAX_DATASETS_PER_USER = 3
MAX_ROWS = 50_000
MAX_COLUMNS = 20


def _run_analysis_in_background(filename: str):
    """
    Runs AnalysisExplainer end-to-end (stats + LLM insights) and stores
    the result to MongoDB via store_dataset_insights — same flow the
    /api/info route would trigger, just fired automatically after upload
    instead of waiting for the user to visit Data Info.
    """
    try:
        explainer = AnalysisExplainer(filename)
        explainer.run()
        logging.info("Background analysis completed for '%s'", filename)
    except Exception as e:
        # Don't crash anything — Data Info will just show empty state
        # and the user (or a future manual trigger) can retry.
        logging.warning("Background analysis failed for '%s': %s", filename, e)


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

        # Fire-and-forget: analysis runs in the background so the upload
        # response isn't delayed by the LLM call. Data Info will show
        # empty state for a few seconds until this finishes.
        thread = threading.Thread(
            target=_run_analysis_in_background,
            args=(file.filename,),
            daemon=True,
        )
        thread.start()

        return {
            "filename": file.filename,
            "rows": rows,
            "columns": cols,
        }

    except ValueError:
        raise

    except Exception as e:
        raise CustomException(e, sys) from e