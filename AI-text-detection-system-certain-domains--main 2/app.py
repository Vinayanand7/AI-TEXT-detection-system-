from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import os
from fordoc import analyze_pdf_document

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file, session
import io
from  ai_detector_pipeline import confidence_score,detect_ai

from models.user_model import create_user, find_user


app = Flask(__name__)
app.secret_key = "3a2d23016e7bf4014a24d90eb81d46ce996ccc866797dc52fb4b37b676dccee4"

MONGO_URI = "mongodb+srv://db_user:OKVAMhKaOFHAcRMM@cluster0.qj5gtmi.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["ai_detector"]
users_collection = db["users"]

bcrypt = Bcrypt(app)
history_collection = db["analysis_history"]
from datetime import datetime
# ------------------------
# HOME ROUTE
# ------------------------
@app.route("/")
def index():
    return redirect(url_for("login"))

# ------------------------
# SIGNUP
# ------------------------
import re
from flask import flash

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        # -------- VALIDATION --------
        if len(username) < 3:
            flash("Username must be at least 3 characters", "error")
            return redirect(url_for("signup"))

        if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
            flash("Password must contain letters and numbers", "error")
            return redirect(url_for("signup"))

        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return redirect(url_for("signup"))

        if find_user(username):
            flash("Username already exists", "error")
            return redirect(url_for("signup"))

        # -------- SAVE USER --------
        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
        create_user(username, hashed_pw)

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# ------------------------
# LOGIN
# ------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        user = find_user(username)

        if not user:
            flash("User not found", "error")
            return redirect(url_for("login"))

        if not bcrypt.check_password_hash(user["password"], password):
            flash("Incorrect password", "error")
            return redirect(url_for("login"))

        session["user"] = username
        flash("Login successful", "success")
        return redirect(url_for("home"))

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    data = list(history_collection.find({"user": session["user"]}).sort("timestamp", -1))

    return render_template("dashboard.html", data=data)

# ------------------------
# HOME PAGE
# ------------------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/text")
def text_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("text.html")


@app.route("/pdf")
def pdf_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("pdf.html")


  # or your pipeline

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ------------------------
# ANALYZE TEXT
# ------------------------
from flask import request, render_template, redirect, url_for, session
from ai_detector_pipeline import detect_ai, confidence_score

@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    # 🔒 protect route
    if "user" not in session:
        return redirect(url_for("login"))

    # 📥 get text from form
    text = request.form.get("text")

    # 🚨 validation
    if not text or not text.strip():
        return "Empty text submitted"

    # 🔥 CALL YOUR CORE LOGIC
    detection = detect_ai(text)
    confidence = confidence_score(detection, text)

    # 📊 structure result (IMPORTANT for report page)
    result = {
        "type": "text",
        "final_score": detection.get("final_score", 0),
        "confidence": confidence,
        "details": detection
    }
    
    app.config["LAST_RESULT"] = result
    history_collection.insert_one({
    "user": session["user"],
    "type": result["type"],
    "result": result,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      })
    # 📄 send to report page
    return render_template("report.html", result=result)





@app.route("/analyze_pdf_local", methods=["POST"])
def analyze_pdf_local():
    if "user" not in session:
        return redirect(url_for("login"))

    pdf_path = request.form.get("pdf_path")

    if not pdf_path:
        return "No path provided"

    # 🔥 Validate file exists
    if not os.path.exists(pdf_path):
        return "File not found"
    
    

    result = analyze_pdf_document(pdf_path)
    app.config["LAST_RESULT"] = result
    history_collection.insert_one({
    "user": session["user"],
    "type": result["type"],
    "result": result,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      })
    return render_template("report.html", result=result)




# ── Add to your imports at top of app.py ──────────────────────
from report_generator import build_text_report, build_pdf_report

# ── Replace your download_report route with this ───────────────
@app.route("/download_report")
def download_report():
    if "user" not in session:
        return redirect(url_for("login"))

    result = app.config.get("LAST_RESULT")
    if not result:
        return "No report available. Please run an analysis first.", 404

    username = session.get("user", "user")
    theme    = request.args.get("theme", "dark")   # ?theme=light / ocean / dark

    if result.get("type") == "text":
        buffer = build_text_report(result, username=username, theme=theme)
    elif result.get("type") == "pdf":
        buffer = build_pdf_report(result, username=username, theme=theme)
    else:
        return "Unknown result type.", 400

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"detectai_report_{theme}.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)