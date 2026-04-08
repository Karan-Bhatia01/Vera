import io
import os
import sys
import pandas as pd
import gridfs
from pymongo import MongoClient

from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        try:
            mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
            client    = MongoClient(mongo_uri)
            self.db   = client["clarityAI_database"]
            self.fs   = gridfs.GridFS(self.db)
            logging.info("MongoDB GridFS connection established")
        except Exception as e:
            raise CustomException(e, sys)

    def store_file(self, file):
        """Store uploaded file in MongoDB GridFS."""
        try:
            self.fs.put(
                file.read(),
                filename=file.filename,
                content_type=file.content_type,
            )
            logging.info("File '%s' stored in GridFS.", file.filename)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _read_csv_robust(file_bytes: bytes) -> pd.DataFrame:
        """
        Try to parse CSV bytes with progressively more lenient settings.
        Picks the parse result with the MOST columns (best delimiter).
        Handles: NULL bytes, encoding issues, inconsistent column counts, bad lines.
        """
        # Strip NULL bytes upfront — C engine still warns on them
        file_bytes = file_bytes.replace(b'\x00', b'')

        # Log raw bytes for diagnosis
        logging.info("File size: %d bytes", len(file_bytes))
        logging.info("First 200 bytes raw: %r", file_bytes[:200])

        encodings  = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        separators = [",", "\t", ";", "|"]
        best_df    = None

        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding=encoding,
                        sep=sep,
                        on_bad_lines="skip",
                        engine="c",
                        skip_blank_lines=True,
                    )
                    # Must have meaningful shape
                    if df.shape[0] < 1 or df.shape[1] < 1:
                        continue

                    # Keep whichever parse yields the most columns
                    if best_df is None or df.shape[1] > best_df.shape[1]:
                        best_df = df
                        logging.info(
                            "CSV candidate — encoding=%s sep=%r shape=%s",
                            encoding, sep, df.shape,
                        )

                except Exception:
                    continue

            # Early exit: good multi-column result found for this encoding
            if best_df is not None and best_df.shape[1] > 1:
                break

        if best_df is None:
            raise ValueError("Could not parse the uploaded file as a valid CSV.")

        logging.info("CSV final parse — shape=%s", best_df.shape)
        return best_df

    def get_preview(self, filename: str = None):
        """
        Reads the latest uploaded CSV file (or a specific one by filename)
        and returns first 5 rows + columns.
        Returns (None, None) if no file exists or parsing fails.
        """
        try:
            # Build query — only look at CSV files
            query = {"filename": {"$regex": r"\.csv$", "$options": "i"}}
            if filename:
                query = {"filename": filename}

            latest = self.fs.find(query).sort("uploadDate", -1).limit(1)

            for file in latest:
                try:
                    file_bytes = file.read()
                    df = self._read_csv_robust(file_bytes)
                    logging.info(
                        "Preview loaded — file=%s shape=%s",
                        file.filename, df.shape,
                    )
                    return df.head().values.tolist(), df.columns.tolist()
                except Exception as parse_err:
                    logging.warning(
                        "Could not parse '%s' for preview: %s",
                        file.filename, parse_err,
                    )
                    return None, None

            # No CSV files found at all
            return None, None

        except Exception as e:
            logging.error("get_preview failed: %s", e)
            return None, None          # Never crash the page — return gracefully

    def get_file_by_name(self, filename: str) -> pd.DataFrame:
        """
        Fetch a specific file from GridFS by filename and return as DataFrame.
        Raises FileNotFoundError if not found.
        """
        try:
            grid_out = self.fs.find_one(
                {"filename": filename},
                sort=[("uploadDate", -1)],
            )
            if grid_out is None:
                raise FileNotFoundError(f"File '{filename}' not found in GridFS.")
            file_bytes = grid_out.read()
            df = self._read_csv_robust(file_bytes)
            logging.info("File '%s' loaded — shape=%s", filename, df.shape)
            return df
        except FileNotFoundError:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def get_all_filenames(self):
        """Return sorted list of CSV filenames only — excludes .pkl and other non-CSV files."""
        try:
            all_files = self.db.fs.files.distinct("filename")
            return sorted(f for f in all_files if f.lower().endswith(".csv"))
        except Exception as e:
            raise CustomException(e, sys)

    def delete_file(self, filename: str) -> bool:
        """
        Delete all GridFS entries matching the given filename.
        Returns True if at least one file was deleted, False if none found.
        """
        try:
            files = list(self.db.fs.files.find({"filename": filename}))
            if not files:
                logging.warning(
                    "Delete requested but '%s' not found in GridFS.", filename
                )
                return False
            for f in files:
                self.fs.delete(f["_id"])
            logging.info(
                "Deleted %d GridFS entry/entries for '%s'.", len(files), filename
            )
            return True
        except Exception as e:
            raise CustomException(e, sys)