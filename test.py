from pymongo import MongoClient
import gridfs

client = MongoClient("....")
db = client["clarityAI_database"]
fs = gridfs.GridFS(db)

for f in fs.find({"filename": "KrishiTwin_Final_Engineered.csv"}):
    fs.delete(f._id)
    print(f"Deleted: {f.filename}")
