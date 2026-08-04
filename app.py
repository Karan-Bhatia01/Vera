import os

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app():

    app = Flask(__name__)
    app.secret_key = os.getenv("SECRET_KEY", "clarityai-secret-key")
    CORS(app)

    from routes.health   import health_bp
    from routes.auth     import auth_bp
    from routes.upload   import upload_bp
    from routes.dataset  import dataset_bp
    from routes.analysis import analysis_bp
    from routes.eda      import eda_bp
    from routes.ml       import ml_bp
 
    app.register_blueprint(health_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(ml_bp)

    return app


app = create_app()

if __name__ == "__main__":
    print("🚀 ClarityAI Flask API running")
    print("Local URL: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health")
    app.run(host="0.0.0.0", port=5000, debug=False)
