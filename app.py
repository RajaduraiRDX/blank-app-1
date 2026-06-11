from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

from aims.processing import analyze_image, analyze_surface_texture, export_csv, export_excel, export_pdf


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "work" / "uploads"
RESULT_DIR = BASE_DIR / "outputs" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(BASE_DIR / "outputs", filename)


@app.post("/api/analyze")
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "Upload an aggregate image first."}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "Upload an aggregate image first."}), 400

    calibration = float(request.form.get("calibration", 0) or 0)
    min_area = float(request.form.get("min_area", 250) or 250)
    threshold_mode = request.form.get("threshold_mode", "otsu")
    multi_particle = request.form.get("multi_particle", "false") == "true"
    analysis_mode = request.form.get("analysis_mode", "morphology")

    safe_name = Path(image.filename).name
    image_path = UPLOAD_DIR / safe_name
    image.save(image_path)

    if analysis_mode == "surface_texture":
        result = analyze_surface_texture(
            image_path=image_path,
            result_dir=RESULT_DIR,
            calibration_mm_per_px=calibration if calibration > 0 else None,
        )
    else:
        result = analyze_image(
            image_path=image_path,
            result_dir=RESULT_DIR,
            calibration_mm_per_px=calibration if calibration > 0 else None,
            min_area_px=min_area,
            threshold_mode=threshold_mode,
            multi_particle=multi_particle,
        )
    return jsonify(result)


@app.get("/download/<run_id>/<kind>")
def download(run_id, kind):
    run_dir = RESULT_DIR / run_id
    measurements = run_dir / "measurements.json"
    if not measurements.exists():
        return jsonify({"error": "Result not found."}), 404

    if kind == "csv":
        path = export_csv(measurements)
    elif kind == "xlsx":
        path = export_excel(measurements)
    elif kind == "pdf":
        path = export_pdf(measurements)
    else:
        return jsonify({"error": "Unknown export type."}), 404

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
