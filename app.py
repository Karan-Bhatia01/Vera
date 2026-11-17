from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)

    from routes.health   import health_bp
    from routes.auth     import auth_bp
    from routes.upload   import upload_bp
    from routes.dataset  import dataset_bp
    from routes.analysis import analysis_bp
    from routes.eda      import eda_bp
    from routes.ml       import ml_bp

    for blueprint in (health_bp, auth_bp, upload_bp, dataset_bp, analysis_bp, eda_bp, ml_bp):
        app.register_blueprint(blueprint)

    return app


app = create_app()

if __name__ == "__main__":
    print("🚀 Vera Flask API running")
    print("Local URL: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health")
    app.run(host="0.0.0.0", port=5000, debug=False)
