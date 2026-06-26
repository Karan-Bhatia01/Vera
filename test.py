from pymongo import MongoClient
import gridfs

client = MongoClient("mongodb+srv://bhatiakaran168_db_user:4d4jWUd0c8qsgrLj@clarityai.fje0nqm.mongodb.net/?appName=clarityAI")
db = client["clarityAI_database"]
fs = gridfs.GridFS(db)

for f in fs.find({"filename": "KrishiTwin_Final_Engineered.csv"}):
    fs.delete(f._id)
    print(f"Deleted: {f.filename}")