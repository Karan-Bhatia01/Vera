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

    def store_file(self, file, owner_email: str = None):
        """Store uploaded file in MongoDB GridFS, tagged with its owner."""
        try:
            self.fs.put(
                file.read(),
                filename=file.filename,
                content_type=file.content_type,
                owner_email=owner_email,
            )
            logging.info("File '%s' stored in GridFS for owner '%s'.", file.filename, owner_email)
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



    def get_all_filenames(self, owner_email: str = None):
        """
        Return sorted list of CSV filenames only — excludes .pkl and other non-CSV files.
        If owner_email is given, only returns files owned by that user.
        """
        try:
            query = {"filename": {"$regex": r"\.csv$", "$options": "i"}}
            if owner_email:
                query["owner_email"] = owner_email

            docs = self.db.fs.files.find(query, {"filename": 1})
            return sorted({d["filename"] for d in docs})
        except Exception as e:
            raise CustomException(e, sys)



    def count_files_for_owner(self, owner_email: str) -> int:
        """Count how many CSV datasets a given user currently has stored."""
        try:
            return self.db.fs.files.count_documents({
                "filename": {"$regex": r"\.csv$", "$options": "i"},
                "owner_email": owner_email,
            })
        except Exception as e:
            raise CustomException(e, sys)
