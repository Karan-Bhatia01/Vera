import sys

from src.core.connections import get_db, get_gridfs
from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        try:
            self.db = get_db()
            self.fs = get_gridfs()
            logging.info("MongoDB GridFS connection established (shared)")
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

    def get_all_filenames(self, owner_email: str = None):
        """
        Return sorted list of CSV filenames only — excludes .pkl and other non-CSV files.
        If owner_email is given, only returns files owned by that user.
        """
        try:
            query = {"filename": {"$regex": r"\.csv$", "$options": "i"}}
            if owner_email:
                query["owner_email"] = owner_email
            return sorted(self.db.fs.files.distinct("filename", query))
        except Exception as e:
            raise CustomException(e, sys)

    def count_files_for_owner(self, owner_email: str) -> int:
        """Count a user's datasets by distinct filename, matching what the dataset list shows."""
        return len(self.get_all_filenames(owner_email))

    def has_file(self, filename: str, owner_email: str = None) -> bool:
        """True if `filename` is stored (for this owner, or for anyone when owner_email is None)."""
        query = {"filename": filename}
        if owner_email:
            query["owner_email"] = owner_email
        try:
            return self.db.fs.files.count_documents(query, limit=1) > 0
        except Exception as e:
            raise CustomException(e, sys)

    def delete_file(self, owner_email: str, filename: str) -> int:
        """Delete every stored copy of `filename` owned by this user. Returns how many were removed."""
        try:
            copies = list(self.db.fs.files.find({"owner_email": owner_email, "filename": filename}, {"_id": 1}))
            for doc in copies:
                self.fs.delete(doc["_id"])
            logging.info("Deleted %d copies of '%s' for owner '%s'.", len(copies), filename, owner_email)
            return len(copies)
        except Exception as e:
            raise CustomException(e, sys)

