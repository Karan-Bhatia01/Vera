# VERA ML Platform - Comprehensive README

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [System Requirements](#system-requirements)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Usage Guide](#usage-guide)
8. [API Documentation](#api-documentation)
9. [Project Structure](#project-structure)
10. [Technology Stack](#technology-stack)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [Support](#support)

---

## Project Overview

**VERA** is an AI-powered data analysis platform that combines scientific computing with Large Language Model reasoning to provide automated, intelligent data insights.

### Philosophy

> "Python does the math. AI does the thinking."

VERA leverages:
- **Python scientific libraries** (NumPy, Pandas, Scikit-Learn) for accurate statistical computation
- **OpenAI-compatible LLM APIs** (OXLO) for contextual reasoning and interpretation
- **MongoDB** for persistent, scalable data storage

### Key Differentiator

Unlike traditional BI tools that rely solely on LLMs for analysis (risking mathematical errors), VERA separates concerns:
- Statistical calculations → Python (accurate, reproducible)
- Interpretation & reasoning → LLM (contextual, actionable)

---

## Features

### 📊 Data Analysis
- **Intelligent Dataset Analysis**: Automatic detection of data quality issues
- **Statistical Insights**: Comprehensive descriptive statistics
- **Missing Value Detection**: Identifies nulls, duplicates, outliers
- **Column-by-Column Analysis**: Individual assessment of each feature
- **Data Preview**: Display first 5 rows with metadata

### 🔧 Data Preprocessing
- **Flexible Cleaning**: Remove duplicates, drop columns, handle nulls
- **AI-Guided Missing Value Strategy**: LLM recommends optimal imputation per column
- **25+ Exploratory Data Analysis Charts**:
  - Distribution plots (histogram, KDE, box plots)
  - Correlation matrices and heatmaps
  - Time series, scatter plots, violin plots
  - And more...
- **Interactive Chart Analysis**: Vision-based AI insights on individual charts

### 🤖 Automated Machine Learning
- **Auto Problem Detection**: Automatically identifies classification vs. regression
- **Feature Engineering**: LLM suggests optimal feature transformations
- **10+ Model Families**:
  - **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, XGBoost, LightGBM, Gradient Boosting
  - **Regression**: Linear Regression, Ridge, Lasso, SVM, Decision Tree, Random Forest, XGBoost, LightGBM, Gradient Boosting
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Model Comparison**: Full metrics comparison across all models
- **Feature Importance**: Automatic calculation and visualization

### 💾 Data Management
- **CSV Upload & Storage**: Persistent storage via MongoDB GridFS
- **Version Tracking**: Multiple datasets, analysis history
- **Model Export**: Download trained models as pickle files
- **Result Caching**: Stored insights and analyses

### 🎨 Modern User Interface
- **Dark Theme Design System**: Professional, modern aesthetic
- **Responsive Layout**: Works on desktop and mobile
- **Smooth Animations**: GPU-accelerated transitions
- **Design Tokens**: Consistent typography, spacing, colors
- **Intuitive Navigation**: Clear workflow through analysis steps

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────┐
│   Web Browser               │
│  (HTML/CSS/JavaScript)      │
└──────────────┬──────────────┘
               │
        HTTP ↓ ↑ JSON/HTML
               │
┌──────────────┴──────────────┐
│   Flask Backend (Python)    │
│  - Route Handling           │
│  - Request Processing       │
│  - Response Formatting      │
└──────────────┬──────────────┘
               │
      Method Calls/APIs
               │
┌──────────────┴──────────────────────────────┐
│   Business Logic Layer                      │
│  ┌─────────────────────────────────────┐   │
│  │ Data Ingestion (file upload)        │   │
│  │ Data Analysis (statistics)          │   │
│  │ EDA Processing (charts)             │   │
│  │ ML Pipeline (model training)        │   │
│  │ Agentic Layer (LLM integration)     │   │
│  └─────────────────────────────────────┘   │
└──────────────┬──────────────────────────────┘
               │
    MongoDB & GridFS (persistence)
    OXLO API (LLM reasoning)
    Matplotlib (visualization)
    Scikit-Learn (ML models)
```

### Data Flow

1. **Upload** → CSV stored in GridFS with filename index
2. **Analyze** → Statistics computed with Pandas → sent to LLM → insights stored
3. **Preprocess** → User configures cleaning → 25 charts generated → base64 encoded
4. **Chart Insights** → User clicks chart → vision API analyzes → insights displayed
5. **ML Pipeline** → Target selected → Features engineered → 10+ models trained → results saved

---

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **MongoDB**: 4.4 or higher (local or Atlas)
- **RAM**: 4GB minimum (8GB recommended for large datasets)
- **Disk**: 500MB for application + space for datasets

### Browser Compatibility

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## Installation & Setup

### Prerequisites

Before starting, ensure you have:

```bash
# Check Python version
python --version  # Should be 3.8+

# Check MongoDB
mongosh --version  # Or verify via MongoDB Compass
```

### Step 1: Clone/Download Repository

```bash
# Option 1: If using git
git clone <repository-url>
cd ClarityAI2.0

# Option 2: If downloaded as zip
unzip ClarityAI2.0.zip
cd ClarityAI2.0
```

### Step 2: Create Python Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation typically takes 2-5 minutes. If you encounter errors, see [Troubleshooting](#troubleshooting).

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required: OXLO API Key (get from https://docs.oxlo.ai)
OXLO_API_KEY

# Required: Flask Secret Key (for sessions)
SECRET_KEY=YOUR_SECRET_KEY

# Optional: MongoDB Connection (defaults to localhost:27017)
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=clarityAI_database
```

### Step 5: Verify MongoDB

```bash
# Start MongoDB (if installed locally)
# Windows: services.msc → MongoDB → right-click → Start
# macOS: brew services start mongodb-community
# Linux: sudo systemctl start mongod

# Verify connection
mongosh localhost:27017
```

### Step 6: Start the Application

```bash
# Windows
python app.py

# macOS/Linux
python3 app.py
```

Expected output:
```
 * Running on http://localhost:5001
 * Press CTRL+C to quit
```

### Step 7: Open in Browser

Navigate to: **[http://localhost:5001](http://localhost:5001)**

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OXLO_API_KEY` | ✅ Yes | None | API key for OXLO LLM service |
| `SECRET_KEY` | ✅ Yes | None | Flask session encryption key |
| `MONGODB_URI` | ❌ No | mongodb://localhost:27017 | MongoDB connection string |
| `MONGODB_DB` | ❌ No | clarityAI_database | Database name |

### Getting OXLO API Key

1. Visit [OXLO Documentation](https://docs.oxlo.ai)
2. Create account and verify email
3. Generate API key in dashboard
4. Add to `.env`: `OXLO_API_KEY=YOUR_OXLO_API_KEY`

### Using MongoDB Atlas (Cloud)

1. Create free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create database user and whitelist IP
3. Get connection string from dashboard
4. Update `.env`:
   ```env
   MONGODB_URI=YOUR_MONGODB_CONNECTION_STRING
   ```

### Customizing Design System

All design tokens are in `static/css/design-system.css`. To change colors:

```css
:root {
  --bg-primary:    #0a0a0f;       /* Change background */
  --accent-primary: #4f46e5;      /* Change main brand color */
  --accent-secondary: #10b981;    /* Change success color */
  /* ... other tokens ... */
}
```

---

## Usage Guide

### Complete Workflow

#### 1️⃣ Upload Data

1. Click **"Upload Dataset"** on navbar
2. Click upload zone or drag-and-drop CSV file
3. Supported formats: `.csv` files, UTF-8 or Latin1 encoding
4. File appears in list below with preview (first 5 rows shown)

**Supported separators**: comma (,), semicolon (;), tab (\t), pipe (|)

#### 2️⃣ Analyze Dataset

1. Click **"Dataset Info"** button next to your file
2. Platform shows:
   - Dataset shape (rows × columns)
   - Null value counts and percentages
   - Unique value counts
   - Numeric statistics (mean, std, min, max)
   - **AI Summary** (generated by LLM)
   - Quality flags and warnings
   - Column-by-column insights
   - Recommended next steps
3. Review AI insights for data quality issues

#### 3️⃣ Configure Preprocessing

1. Click **"Preprocess"** from dataset info page
2. Select columns to drop (unchecked = keep)
3. For each column with missing values, choose strategy:
   - **Drop**: Remove rows with nulls
   - **Mean** (numeric): Fill with average value
   - **Median** (numeric): Fill with middle value
   - **Mode** (categorical): Fill with most common
   - **Forward Fill**: Use previous value
4. Click **"Get AI Suggestions"** for LLM recommendations
5. Submit form to execute preprocessing

#### 4️⃣ Review EDA Charts

1. Platform generates 25+ charts showing:
   - Distributions of each numeric column
   - Correlation heatmap
   - Top categorical values
   - Missing value patterns
   - Outlier detection
   - And more...
2. Scroll through charts
3. **Click any chart → "AI Insights"** to get vision-based analysis
4. AI explains what the chart shows, key findings, and anomalies

#### 5️⃣ Train ML Models

1. Click **"Start ML Pipeline"**
2. Select **Target Column** from dropdown (the column you want to predict)
3. Click **"Train Models"** to start training
4. Platform automatically:
   - Detects problem type (classification/regression)
   - Engineers features using LLM guidance
   - Trains 10+ model families
   - Tunes hyperparameters
   - Selects best model
5. View results:
   - Best model name and metric
   - Comparison table of all models
   - Confusion matrix (classification) or residuals (regression)
   - Feature importance chart
   - Training time

#### 6️⃣ Download Model

1. On ML results page, find **"Download Model"** button
2. Click to download `.pkl` file (Python pickle format)
3. Use in own scripts:
   ```python
   import pickle
   model = pickle.load(open('model.pkl', 'rb'))
   predictions = model.predict(X_new)
   ```

---

## API Documentation

### Base URL

```
http://localhost:5001
```

### Authentication

No authentication required (production: add API key validation)

### Endpoints

#### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "connected",
  "mongodb": true,
  "timestamp": "2026-04-04T15:00:00Z"
}
```

---

#### Upload File

```http
GET /upload
POST /upload
```

**POST Body** (multipart/form-data):
```
file: <CSV file>
```

**Response:**
- GET: HTML form + file list
- POST: Redirect to `/upload?filename=<filename>`

---

#### Get Dataset Info

```http
GET /info?filename=<filename>
```

**Response:** HTML page with:
- Dataset statistics (shape, nulls, unique values)
- AI-generated insights
- Quality flags
- Column analysis
- Next steps

---

#### List Stored Datasets

```http
GET /api/stored_datasets
```

**Response:**
```json
{
  "datasets": [
    {
      "filename": "Churn_Modelling.csv",
      "stored_at": "2026-04-04T15:00:00Z",
      "shape": [10000, 12],
      "summary": "Dataset quality is good..."
    }
  ]
}
```

---

#### Get Stored Insights

```http
GET /api/insights/<filename>
```

**Response:**
```json
{
  "_id": "ObjectId(...)",
  "filename": "Churn_Modelling.csv",
  "analysis": {
    "shape": [10000, 12],
    "columns": ["id", "CustomerId", ...],
    "null_values": {"CustomerId": 0, ...},
    "unique_counts": {...}
  },
  "ai_insights": {
    "summary": "High quality dataset...",
    "quality_flags": ["flag1", "flag2"],
    "column_insights": {...},
    "next_steps": ["step1", "step2"]
  }
}
```

---

#### Preprocessing

```http
GET /preprocessing?filename=<filename>
POST /preprocessing
```

**GET:** Shows preprocessing form

**POST Body:**
```json
{
  "filename": "Churn_Modelling.csv",
  "columns_to_drop": ["id", "CustomerId"],
  "missing_strategies": {
    "Age": "mean",
    "Salary": "median",
    "Status": "mode"
  }
}
```

**Response:** HTML page with EDA charts

---

#### Chart Analysis

```http
POST /analyse_chart
```

**Request Body:**
```json
{
  "image_b64": "iVBORw0KGgoAAAANS...",
  "chart_title": "Age Distribution",
  "filename": "Churn_Modelling.csv"
}
```

**Response:**
```json
{
  "represents": "The plot shows the distribution of customer ages...",
  "key_findings": [
    "Most customers are between 25-50 years old",
    "Slight skew towards older customers"
  ],
  "anomalies": [
    "Outlier: 2 customers over 100 years old"
  ],
  "recommendations": [
    "Consider age groups for categorical analysis",
    "Investigate outliers for data quality"
  ]
}
```

---

#### Machine Learning

```http
GET /ml?filename=<filename>
POST /ml
```

**GET:** Shows target column selection form

**POST Body:**
```json
{
  "filename": "Churn_Modelling.csv",
  "target_column": "Exited"
}
```

**Response:** HTML page with ML results

---

#### Download Model

```http
GET /download_model?doc_id=<id>&model_name=<name>
```

**Response:** `.pkl` file (binary pickle format)

---

#### List Models for Dataset

```http
GET /api/models/<filename>
```

**Response:**
```json
{
  "filename": "Churn_Modelling.csv",
  "models": [
    "LogisticRegression",
    "RandomForest",
    "XGBoost"
  ]
}
```

---

## Project Structure

```
ClarityAI2.0/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create yourself)
├── TECHNICAL_REPORT.txt           # Detailed technical documentation
├── WORKFLOW.txt                   # User workflow guide
│
├── src/                           # Backend Python modules
│   ├── __init__.py
│   ├── logger.py                 # Logging configuration
│   ├── exception.py              # Custom exception handling
│   ├── utils.py                  # Utility functions
│   ├── api_routes.py             # API utilities
│   │
│   ├── agenticLayer/
│   │   └── llm.py                # LLM integration (AnalysisExplainer)
│   │
│   └── components/
│       ├── __init__.py
│       ├── data_ingestion.py     # CSV upload & storage
│       ├── data_info.py          # Dataset statistics
│       ├── eda_processing.py     # Data preprocessing & charts
│       ├── ml_pipeline.py        # Model training
│       ├── mongo_storage.py      # MongoDB operations
│       ├── notebook_exporter.py  # Jupyter notebook generation
│       └── rag_pipeline.py       # Semantic search (future)
│
├── frontend/                       # Next.js Frontend (alternative)
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── src/
│   └── public/
│
├── templates/                      # Jinja2 HTML templates
│   ├── vera_landing.html          # Landing page
│   ├── index.html                 # Upload page
│   ├── info.html                  # Dataset analysis
│   ├── eda_processing.html        # Preprocessing config
│   ├── preprocessing_result.html  # EDA charts
│   ├── ml_input.html              # ML target selection
│   └── ml_results.html            # ML results display
│
├── static/                         # Static assets
│   ├── css/
│   │   ├── design-system.css      # Design tokens & variables
│   │   ├── layout.css             # Page layouts
│   │   ├── components.css         # UI components
│   │   ├── vera_landing.css       # Landing page styles
│   │   ├── info.css               # Analysis page styles
│   │   ├── eda.css                # EDA page styles
│   │   ├── ml.css                 # ML results styles
│   │   ├── chatbot.css            # Chat widget styles
│   │   └── modern-ui.js           # Interactive JavaScript
│   │
│   └── js/
│       ├── vera-chatbot.js        # Chat functionality
│       └── modern-ui.js           # UI interactions
│
├── logs/                          # Application logs
│   └── [timestamp].log            # Timestamped log files
│
└── notebooks/                     # Jupyter notebooks
    └── test.ipynb                 # Example notebook
```

---

## Technology Stack

### Backend

| Technology | Purpose | Version |
|-----------|---------|---------|
| Flask | Web framework | 2.x |
| Python | Programming language | 3.8+ |
| MongoDB | Document database | 4.4+ |
| PyMongo | MongoDB driver | 3.x+ |
| NumPy | Numerical computing | 1.20+ |
| Pandas | Data manipulation | 1.2+ |
| Scikit-Learn | ML models & preprocessing | 0.24+ |
| Matplotlib | Chart generation | 3.3+ |
| Seaborn | Statistical plotting | 0.11+ |
| XGBoost | Gradient boosting | 1.3+ |
| OpenAI SDK | LLM API client | 0.27+ |
| python-dotenv | Environment variables | 0.19+ |

### Frontend

| Technology | Purpose |
|-----------|---------|
| HTML5 | Semantic markup |
| CSS3 | Styling & animations |
| JavaScript (Vanilla) | Interactivity |
| Jinja2 | Template engine |
| CSS Grid | Responsive layout |
| CSS Variables | Design tokens |

### Infrastructure

| Technology | Purpose |
|-----------|---------|
| MongoDB GridFS | Large file storage |
| OXLO API | LLM reasoning |
| Git | Version control |

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: "OXLO_API_KEY not configured"

**Cause**: Missing `.env` file or key not set

**Solution**:
1. Create `.env` file in project root
2. Add: `OXLO_API_KEY=sk_live_your_actual_key`
3. Restart Flask application
4. Refresh browser

---

#### Issue: MongoDB Connection Failed

**Cause**: MongoDB not running or wrong connection string

**Solution**:
```bash
# Windows: Check Services
services.msc → look for MongoDB → right-click Start

# macOS
brew services start mongodb-community

# Linux
sudo systemctl start mongod

# Verify connection
mongosh localhost:27017
```

If using MongoDB Atlas:
1. Update `.env`: `MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/`
2. Ensure IP is whitelisted in Atlas dashboard
3. Restart Flask

---

#### Issue: Dependencies Won't Install

**Cause**: Python version incompatible, pip outdated, or network issue

**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Try specific versions
pip install flask==2.3.0 pymongo==3.12.3

# Clear cache and retry
pip cache purge
pip install -r requirements.txt
```

---

#### Issue: Chart Analysis Returns "Retry" Button

**Cause**: OXLO API key invalid, rate limited, or API error

**Solution**:
1. Check browser console (F12 → Console)
2. Check Flask logs (terminal showing Flask output)
3. Verify `OXLO_API_KEY` is correct
4. Wait 1 minute (rate limit)
5. Verify internet connectivity
6. Try simpler chart first

---

#### Issue: File Upload Fails

**Cause**: File format, encoding, or size issue

**Solution**:
1. Ensure file is `.csv` format
2. Open CSV in text editor, check encoding (should be UTF-8)
3. Check file size (< 50MB recommended)
4. Verify columns are comma-separated
5. Try with sample dataset first

---

#### Issue: ML Pipeline Takes Very Long or Crashes

**Cause**: Large dataset, many features, or memory issue

**Solution**:
1. Reduce dataset size (remove rows)
2. Drop unnecessary columns
3. Reduce target class imbalance
4. Check available RAM (`Task Manager` on Windows)
5. Increase virtual memory if needed

---

#### Issue: Port 5001 Already in Use

**Cause**: Another application using port 5001

**Solution**:
```bash
# Windows: Find what's using port 5001
netstat -ano | findstr :5001

# Kill the process
taskkill /PID <PID> /F

# Or use different port
# Modify app.py: app.run(debug=True, port=5002)
```

---

### Debug Checklist

When something isn't working:

- [ ] Check `.env` file exists and has `OXLO_API_KEY`
- [ ] Verify MongoDB running: `mongosh localhost:27017`
- [ ] Check Flask terminal for errors
- [ ] Open browser console: F12 → Console tab
- [ ] Look for yellow/red errors
- [ ] Check Network tab for failed API calls
- [ ] Verify file format (CSV, UTF-8)
- [ ] Check logs: `logs/[date].log`
- [ ] Try simpler operation first
- [ ] Restart Flask application
- [ ] Clear browser cache (Ctrl+Shift+Delete)

---

## Contributing

### How to Contribute

1. **Report Bugs**: Open issue with reproduction steps
2. **Suggest Features**: Describe use case and benefits
3. **Submit Code**: 
   - Fork repository
   - Create feature branch
   - Commit changes with clear messages
   - Push and create pull request

### Development Guidelines

- Follow PEP 8 for Python code
- Add docstrings to functions
- Test changes before submitting
- Update documentation as needed
- Use descriptive branch names: `feature/xyz` or `fix/bug-name`

### Setting Up Development Environment

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black pylint

# Format code
black src/

# Run linter
pylint src/

# Run tests
pytest
```

---

## Support

### Getting Help

1. **Check Documentation**
   - Read TECHNICAL_REPORT.txt for detailed architecture
   - Review this README for common issues
   - Check WORKFLOW.txt for user guide

2. **Search Issues** 
   - GitHub Issues for reported problems
   - Stack Overflow for Python/MongoDB questions

3. **Contact Support**
   - Email: support@example.com
   - Documentation: https://docs.clarityai.example.com
   - Discord: [Community Server](https://discord.gg/example)

### Useful Links

- [Flask Documentation](https://flask.palletsprojects.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [OXLO API Documentation](https://docs.oxlo.ai)
- [Python Pickle Format](https://docs.python.org/3/library/pickle.html)

---

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## Changelog

### Version 1.0 (Current)
- ✅ Core data analysis features
- ✅ ML pipeline with 10+ models
- ✅ LLM integration (OXLO)
- ✅ Data preprocessing & EDA
- ✅ Model export & download
- ✅ Modern UI with design system

### Planned Features
- 🔄 Jupyter notebook generation
- 🔄 Advanced RAG pipeline
- 🔄 Real-time progress WebSockets
- 🔄 Multi-user collaboration
- 🔄 Model versioning & tracking
- 🔄 Auto ML with genetic algorithms

---

## FAQ

**Q: Can I use my own LLM instead of OXLO?**
A: Yes! Modify `src/agenticLayer/llm.py` to use your provider (OpenAI, Anthropic, etc.). Just change the API endpoint and authentication.

**Q: Does VERA store my data permanently?**
A: Yes, in MongoDB. You can delete datasets via MongoDB CLI or UI. No automatic cleanup occurs.

**Q: Can I deploy VERA to production?**
A: Yes! Use Gunicorn for Flask and MongoDB Atlas for cloud database. Add authentication, SSL, and rate limiting. See deployment guide.

**Q: What's the maximum file size?**
A: GridFS supports up to 16MB per chunk, unlimited total. Practical limit ~1GB based on RAM. Larger files may timeout processing.

**Q: How do I backup my data?**
A: Use `mongodump` to backup MongoDB collections. Store backups separately.

**Q: Can I train on GPU?**
A: Yes! XGBoost and scikit-learn support GPU acceleration. Install CUDA and update requirements.txt.

---

## Acknowledgments

Built with ❤️ using:
- Open-source ML libraries (NumPy, Pandas, Scikit-Learn)
- Flask web framework
- MongoDB database
- OXLO LLM API

---

**Last Updated**: April 4, 2026  
**Status**: Production Ready  
**Maintainer**: ClarityAI Team

---

## Quick Start Command

```bash
# One-command setup (assumes MongoDB running)
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt && \
python app.py
```

Then visit: **http://localhost:5001**

---

**Happy analyzing! 🚀**
