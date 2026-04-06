import io
import os
import json
import threading

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, session, send_file, Response,
)
from src.components.data_ingestion import DataIngestion
from src.agenticLayer.llm import AnalysisExplainer
from src.components.job_store import create_job, update_job, get_job

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-key-change-in-production")

# ── Singleton data_ingestion (avoid recreating MongoClient every request) ─────
data_ingestion = DataIngestion()


# ── Health Check ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    try:
        from pymongo import MongoClient
        MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1500).admin.command("ping")
        return jsonify({"status": "connected", "mongodb": True})
    except Exception:
        return jsonify({"status": "offline", "mongodb": False}), 503


# ── Landing ────────────────────────────────────────────────────────────────────
@app.route("/")
def vera_landing():
    return render_template("vera_landing.html")


# ── Upload ─────────────────────────────────────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def index():
    selected_filename = request.args.get("filename")

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            data_ingestion.store_file(file)
            return redirect(url_for("index", filename=file.filename))

    filenames               = data_ingestion.get_all_filenames()
    preview_data, columns   = data_ingestion.get_preview()

    return render_template(
        "index.html",
        filenames=filenames,
        selected_filename=selected_filename,
        preview_data=preview_data,
        columns=columns,
    )


@app.route("/info")
def info_layer():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("index"))

    ai_explainer = AnalysisExplainer(filename=filename)
    ai_result    = ai_explainer.run()

    # Store insights in MongoDB (non-blocking thread)
    def _store_insights():
        try:
            from src.components.mongo_storage import store_dataset_insights
            store_dataset_insights(
                filename    = filename,
                analysis    = ai_result["analysis"],
                ai_insights = ai_result["ai_insights"],
                unique      = ai_result["unique"],
            )
            app.logger.info(f"✅ Stored insights for {filename}")
        except Exception as e:
            app.logger.error(f"❌ Failed to store insights: {e}")

    thread = threading.Thread(target=_store_insights, daemon=False)
    thread.start()

    return render_template(
        "info.html",
        analysis    = ai_result["analysis"],
        unique      = ai_result["unique"],
        ai_insights = ai_result["ai_insights"],
        filename    = filename,
    )


# ── Stored Datasets API ────────────────────────────────────────────────────────
@app.route("/api/stored_datasets")
def api_stored_datasets():
    try:
        from src.components.mongo_storage import list_stored_datasets
        return jsonify({"datasets": list_stored_datasets()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Stored Insights API ────────────────────────────────────────────────────────
@app.route("/api/insights/<path:filename>")
def api_get_insights(filename):
    try:
        from src.components.mongo_storage import get_dataset_insights
        doc = get_dataset_insights(filename)
        if not doc:
            return jsonify({"error": "Not found"}), 404
        return jsonify(doc)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Preprocessing + EDA ────────────────────────────────────────────────────────
@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing_inputs():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("index"))

    from src.components.eda_processing import DataPreprocessing
    dp = DataPreprocessing(filename=filename, target_column="",
                           oxlo_api_key=os.environ.get("OXLO_API_KEY", ""))

    if request.method == "GET":
        ai_strategy               = dp.get_ai_insights()
        previous_target           = session.get(f"prev_target_{filename}", "")
        previous_columns_to_drop  = session.get(f"prev_cols_drop_{filename}", [])
        previous_missing_strategy = session.get(f"prev_missing_{filename}", {})
        return render_template(
            "eda_processing.html",
            filename=filename,
            columns=dp.df.columns.tolist(),
            null_counts=dp.df.isnull().sum().to_dict(),
            ai_strategy=ai_strategy,
            has_nulls=bool(dp.df.isnull().values.any()),
            shape=dp.df.shape,
            previous_target=previous_target,
            previous_columns_to_drop=previous_columns_to_drop,
            previous_missing_strategy=previous_missing_strategy,
        )

    target_column   = request.form.get("target_column", "")
    columns_to_drop = request.form.getlist("columns_to_drop")
    missing_value_strategy = {}
    for col in dp.df.columns:
        strategy = request.form.get(f"missing_{col}")
        if strategy:
            missing_value_strategy[col] = strategy

    session[f"prev_target_{filename}"]    = target_column
    session[f"prev_cols_drop_{filename}"] = columns_to_drop
    session[f"prev_missing_{filename}"]   = missing_value_strategy
    session.modified = True

    dp.target_column   = target_column
    dp.columns_to_drop = columns_to_drop
    cleaned_df = dp.preprocess_data(missing_value_strategy=missing_value_strategy)
    eda_report = dp.generate_eda_report(cleaned_df)

    return render_template(
        "preprocessing_result.html",
        filename=filename,
        shape=cleaned_df.shape,
        preview_data=cleaned_df.head(10).to_dict(orient="records"),
        columns=cleaned_df.columns.tolist(),
        eda_report=eda_report,
    )


# ── Chart AI Analysis ──────────────────────────────────────────────────────────
@app.route("/analyse_chart", methods=["POST"])
def analyse_chart_route():
    data        = request.get_json()
    image_b64   = data.get("image_b64", "")
    chart_title = data.get("chart_title", "")
    filename    = data.get("filename", "")

    if not image_b64 or not chart_title:
        return jsonify({"error": "Missing image_b64 or chart_title"}), 400

    from src.utils import analyse_chart, empty_analysis
    oxlo_key   = os.environ.get("OXLO_API_KEY", "")
    oxlo_url   = "https://api.oxlo.ai/v1/chat/completions"
    oxlo_model = "mistral-7b"

    try:
        result = analyse_chart(image_b64, chart_title, oxlo_key, oxlo_url, oxlo_model)
        if filename:
            try:
                from src.components.mongo_storage import store_chart_insight
                store_chart_insight(filename, chart_title, result)
            except Exception:
                pass
        return jsonify(result)
    except Exception:
        return jsonify(empty_analysis(chart_title)), 200


# ── ML Pipeline ────────────────────────────────────────────────────────────────
@app.route("/ml", methods=["GET", "POST"])
def ml_pipeline_route():
    filename = request.args.get("filename")
    if not filename:
        return redirect(url_for("index"))

    if request.method == "GET":
        from src.utils import load_dataframe_from_mongo
        df = load_dataframe_from_mongo(filename)
        return render_template("ml_input.html", filename=filename, columns=df.columns.tolist())

    target_column = request.form.get("target_column", "")
    if not target_column:
        return redirect(url_for("ml_pipeline_route", filename=filename))

    job_id = create_job()

    def run_pipeline():
        try:
            update_job(job_id, status="running", progress=0, message="Starting...")
            from src.components.ml_pipeline import MLPipeline
            pipeline = MLPipeline(filename=filename, target_column=target_column)
            output   = pipeline.run(
                progress_callback=lambda pct, msg: update_job(job_id, progress=pct, message=msg)
            )
            update_job(job_id, status="done", progress=100, result=output)
        except Exception as e:
            update_job(job_id, status="error", error=str(e), message=f"Pipeline failed: {e}")

    threading.Thread(target=run_pipeline, daemon=True).start()
    return render_template("ml_training.html", job_id=job_id, filename=filename,
                           target_column=target_column)


# ── Model Download ─────────────────────────────────────────────────────────────
@app.route("/download_model")
def download_model():
    doc_id     = request.args.get("doc_id", "")
    model_name = request.args.get("model_name", "")
    if not doc_id or not model_name:
        return jsonify({"error": "Missing doc_id or model_name"}), 400
    try:
        from bson import ObjectId
        from pymongo import MongoClient
        import gridfs
        client    = MongoClient("mongodb://localhost:27017/")
        db        = client["clarityAI_database"]
        fs        = gridfs.GridFS(db)
        col       = db["ml_results"]
        doc       = col.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "ML result not found"}), 404
        gridfs_id = doc.get("model_gridfs_ids", {}).get(model_name)
        if not gridfs_id:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        pkl_bytes     = fs.get(ObjectId(gridfs_id)).read()
        filename_stem = doc.get("filename", "model").replace(".csv", "")
        download_name = f"{filename_stem}__{model_name}.pkl"
        return send_file(io.BytesIO(pkl_bytes), mimetype="application/octet-stream",
                         as_attachment=True, download_name=download_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── List models ────────────────────────────────────────────────────────────────
@app.route("/api/models/<path:filename>")
def api_list_models(filename):
    try:
        from pymongo import MongoClient
        db  = MongoClient("mongodb://localhost:27017/")["clarityAI_database"]
        doc = db["ml_results"].find_one({"filename": filename}, sort=[("_id", -1)])
        if not doc:
            return jsonify({"models": [], "doc_id": None})
        return jsonify({
            "models":       list(doc.get("model_gridfs_ids", {}).keys()),
            "doc_id":       str(doc["_id"]),
            "best_model":   doc.get("best_model", ""),
            "problem_type": doc.get("problem_type", ""),
            "target":       doc.get("target_column", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Notebook Download ──────────────────────────────────────────────────────────
@app.route("/download_notebook")
def download_notebook():
    filename      = request.args.get("filename", "")
    target_column = request.args.get("target", "")
    if not filename:
        return jsonify({"error": "Missing filename"}), 400
    try:
        from src.components.notebook_exporter import generate_notebook
        notebook      = generate_notebook(filename=filename, target_column=target_column)
        nb_json       = json.dumps(notebook, indent=2)
        download_name = filename.replace(".csv", "") + "_pipeline.ipynb"
        return Response(nb_json, mimetype="application/x-ipynb+json",
                        headers={"Content-Disposition": f'attachment; filename="{download_name}"'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Chat API ───────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data     = request.get_json() or {}
    query    = data.get("query", "").strip()
    filename = data.get("filename", None)  # Dataset context
    history  = data.get("history", [])
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        from src.components.rag_pipeline import chat
        result = chat(query=query, filename=filename, history=history)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({
            "answer": f"Error: {str(e)[:100]}",
            "dataset": filename,
            "context_available": False,
        }), 200


# ── Generate ML Summary ────────────────────────────────────────────────────────
@app.route("/api/generate_summary", methods=["POST"])
def api_generate_summary():
    data = request.get_json() or {}
    filename = data.get("filename", "").strip()
    best_model = data.get("best_model", "").strip()
    problem_type = data.get("problem_type", "").strip()
    target_column = data.get("target_column", "").strip()
    
    if not all([filename, best_model, problem_type, target_column]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("OXLO_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OXLO_API_KEY", "not-needed")
        )
        
        # Create a detailed structured summary prompt
        summary_prompt = f"""
        Provide a comprehensive, well-structured summary of the following ML pipeline results:
        
        Dataset: {filename}
        Best Model: {best_model}
        Problem Type: {problem_type}
        Target Column: {target_column}
        
        Structure your response as follows:
        
        📊 MODEL OVERVIEW
        [Brief description of the model and its purpose]
        
        🎯 KEY PERFORMANCE METRICS
        [Main metrics and their significance]
        
        💡 KEY INSIGHTS
        - [Insight 1]
        - [Insight 2]
        - [Insight 3]
        
        ⚠️ IMPORTANT CONSIDERATIONS
        [Edge cases, limitations, or important considerations]
        
        ✅ RECOMMENDATIONS
        [Next steps or recommendations for improvement]
        
        Be professional but accessible. Include specific details about what makes this model effective.
        """
        
        response = client.chat.completions.create(
            model="deepseek-r1-8b",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.7,
            max_tokens=1200,
            timeout=15
        )
        
        summary = response.choices[0].message.content.strip()
        return jsonify({"summary": summary})
    
    except Exception as e:
        app.logger.warning(f"Summary generation via OXLO API failed: {e}")
        
        # Fallback: Generate a detailed structured summary without API
        fallback_summary = f"""📊 MODEL OVERVIEW
Trained {best_model} model for {problem_type} prediction on {filename} dataset with {target_column} as the target variable.

🎯 KEY PERFORMANCE METRICS
This model has been optimized for the {problem_type} task. Review the detailed metrics table above for accuracy, precision, recall, and other relevant performance indicators specific to this task.

💡 KEY INSIGHTS
- The model was trained using automated feature engineering with selected numeric and categorical features
- Performance metrics are displayed in the Model Comparison table above
- Feature importance analysis shows the top contributing features to model predictions

⚠️ IMPORTANT CONSIDERATIONS
- Ensure input data matches the training data format and feature types
- Monitor model performance on new data in production
- Consider periodic retraining as data distributions change

✅ RECOMMENDATIONS
- Review the Feature Engineering Plan section to understand data transformations
- Check SHAP explainability section for detailed feature impact analysis
- Export the model for deployment or use in production pipelines"""
        
        return jsonify({
            "summary": fallback_summary,
            "generated_via": "fallback"
        }), 200


# ── Task Status ────────────────────────────────────────────────────────────────
@app.route("/api/task_status/<job_id>")
def api_task_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "message":  job["message"],
        "error":    job["error"],
        "done":     job["status"] == "done",
    })


# ── ML Results from Job ────────────────────────────────────────────────────────
@app.route("/ml_results/<job_id>")
def ml_results_from_job(job_id):
    job = get_job(job_id)
    if not job or job["status"] != "done":
        return redirect(url_for("index"))
    output = job["result"]
    return render_template(
        "ml_results.html",
        filename      = output["filename"],
        target_column = output["target_column"],
        problem_type  = output["problem_type"],
        feature_plan  = output["feature_plan"],
        results       = output["results"],
        best_model    = output["best_model"],
        rank_metric   = output["rank_metric"],
        mongo_doc_id  = output["mongo_doc_id"],
        shap          = output.get("shap", {}),
    )


# ── SHAP API ───────────────────────────────────────────────────────────────────
@app.route("/api/shap/<doc_id>")
def api_shap(doc_id):
    try:
        from bson import ObjectId
        from pymongo import MongoClient
        db  = MongoClient("mongodb://localhost:27017/")["clarityAI_database"]
        doc = db["ml_results"].find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Not found"}), 404
        shap = doc.get("shap", {})
        return jsonify({
            "top_features": shap.get("top_features", []),
            "model_name":   shap.get("model_name", ""),
            "available":    bool(shap.get("summary_plot")),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Rebuild RAG (deprecated) ────────────────────────────────────────────────
@app.route("/api/rebuild_rag", methods=["POST"])
def api_rebuild_rag():
    """Deprecated endpoint - FAISS no longer used."""
    return jsonify({
        "status": "ok",
        "message": "FAISS is deprecated. Database-direct chatbot is now used.",
        "chunks": 0
    })


# ── RAG Status (deprecated) ─────────────────────────────────────────────────
@app.route("/api/rag_status", methods=["GET"])
def api_rag_status():
    """Deprecated endpoint - FAISS no longer used."""
    return jsonify({
        "status": "deprecated",
        "message": "FAISS is no longer used. Chatbot uses MongoDB directly.",
        "indexes": []
    })


# ── Test Chat ───────────────────────────────────────────────────────────────
@app.route("/api/test_chat", methods=["POST"])
def api_test_chat():
    """Test the chatbot with a sample dataset."""
    try:
        from src.components.rag_pipeline import chat
        
        # Test with Churn dataset if available
        result = chat(query="Tell me about this dataset", filename="Churn_Modelling.csv")
        
        return jsonify({
            "status": "ok",
            "test_query": "Tell me about this dataset",
            "response": result,
        })
    except Exception as e:
        app.logger.error(f"Chat test failed: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)