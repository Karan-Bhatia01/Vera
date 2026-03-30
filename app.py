import os

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)
from src.components.data_ingestion import DataIngestion
from src.agenticLayer.llm import AnalysisExplainer

app = Flask(__name__)

data_ingestion = DataIngestion()


# Home / Upload 
@app.route("/", methods=["GET", "POST"])
def index():
    selected_filename = request.args.get("filename")

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            data_ingestion.store_file(file)
            return redirect(url_for("index", filename=file.filename))

    filenames = data_ingestion.get_all_filenames()
    preview_data, columns = data_ingestion.get_preview()

    return render_template(
        "index.html",
        filenames=filenames,
        selected_filename=selected_filename,
        preview_data=preview_data,
        columns=columns
    )


# Dataset Info + AI Insights
@app.route("/info")
def info_layer():
    filename = request.args.get("filename")

    if not filename:
        return redirect(url_for("index"))

    ai_explainer = AnalysisExplainer(filename=filename)
    ai_result = ai_explainer.run()

    return render_template(
        "info.html",
        analysis=ai_result["analysis"],
        unique=ai_result["unique"],
        ai_insights=ai_result["ai_insights"],
    )


# Preprocessing + EDA 
@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing_inputs():
    filename = request.args.get("filename")

    if not filename:
        return redirect(url_for("index"))

    from src.components.eda_processing import DataPreprocessing

    dp = DataPreprocessing(
        filename=filename,
        target_column="",
        oxlo_api_key=os.environ.get("OXLO_API_KEY", ""),
    )

    # GET: show form with AI-suggested fill strategies
    if request.method == "GET":
        ai_strategy = dp.get_ai_insights()   # {col: method} or {} if no nulls

        return render_template(
            "eda_processing.html",
            filename=filename,
            columns=dp.df.columns.tolist(),
            null_counts=dp.df.isnull().sum().to_dict(),
            ai_strategy=ai_strategy,
            has_nulls=bool(dp.df.isnull().values.any()),
            shape=dp.df.shape,
        )

    # POST: preprocess : generate EDA charts → render report
    target_column   = request.form.get("target_column", "")
    columns_to_drop = request.form.getlist("columns_to_drop")

    missing_value_strategy = {}
    for col in dp.df.columns:
        strategy = request.form.get(f"missing_{col}")
        if strategy:
            missing_value_strategy[col] = strategy

    dp.target_column  = target_column
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)