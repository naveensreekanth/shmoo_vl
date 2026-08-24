"""
Flask Backend API Server
------------------------
Serves API endpoints for file upload, ML model training & execution, plot rendering,
and downloadable PDF report generation with text options (LLM vs Template).
"""

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os, json, uuid, traceback
import pandas as pd
import numpy as np
from pathlib import Path

from data.preprocessor import ShmooPreprocessor
from ml.model import ShmooModel
from report.generator import ReportGenerator
from report.plot_builder import build_shmoo_plot
from text.llm_engine import LLMEngine
from text.template_engine import TemplateEngine

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

sessions = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        session_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix
        save_path = UPLOAD_DIR / f"{session_id}{ext}"
        file.save(save_path)

        # Preprocess data
        preprocessor = ShmooPreprocessor()
        meta = preprocessor.load(str(save_path))
        df = preprocessor.process()

        # Train ML model
        model = ShmooModel()
        results = model.train_and_evaluate(df)

        # Save web dark-theme plot PNG to disk for high-reliability static serving
        web_plot_path = UPLOAD_DIR / f"{session_id}_web.png"
        build_shmoo_plot(df, results, save_path=str(web_plot_path), as_base64=True)

        sessions[session_id] = {
            'df': df,
            'preprocessor': preprocessor,
            'model': model,
            'meta': meta,
            'results': results,
            'save_path': str(save_path),
            'web_plot_path': str(web_plot_path)
        }

        # Serialize plot points for interactive browser chart with tooltips
        die_col = 'Die_ID' if 'Die_ID' in df.columns else ('Test_ID' if 'Test_ID' in df.columns else None)
        points_data = []
        for _, row in df.iterrows():
            points_data.append({
                'vdd': round(float(row['VDD_V']), 4),
                'freq': round(float(row['Frequency_GHz']), 4),
                'result': str(row['Test_Result']),
                'code': str(row['Failure_Code']),
                'die': str(row[die_col]) if die_col else 'Device 1',
                'pat': str(row.get('Pattern_ID', row.get('March_Algorithm', ''))) if ('Pattern_ID' in df.columns or 'March_Algorithm' in df.columns) else ''
            })

        return jsonify({
            'session_id': session_id,
            'meta': meta,
            'results': _serialize_results(results),
            'plot_data': {
                'points': points_data,
                'vdd_range': [float(df['VDD_V'].min()), float(df['VDD_V'].max())],
                'freq_range': [float(df['Frequency_GHz'].min()), float(df['Frequency_GHz'].max())],
            },
            'plot_url': f'/api/plot/{session_id}.png'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/plot/<session_id>.png')
def get_plot_image(session_id):
    try:
        web_plot_path = UPLOAD_DIR / f"{session_id}_web.png"
        if not web_plot_path.exists():
            session = sessions.get(session_id)
            if not session:
                return jsonify({'error': 'Session not found'}), 404
            build_shmoo_plot(session['df'], session['results'], save_path=str(web_plot_path), as_base64=True)

        return send_file(str(web_plot_path), mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report', methods=['POST'])
def generate_report():
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        text_mode = data.get('text_mode', 'template')  # 'llm' or 'template'
        test_methodology = data.get('test_methodology', 'MBIST')  # 'MBIST', 'LBIST', or 'ATPG'

        session = sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        results = session['results']
        meta = session['meta']
        df = session['df']

        # Generate text narrative
        if text_mode == 'llm':
            engine = LLMEngine()
            if engine.is_available():
                narrative = engine.generate(results, meta, test_methodology=test_methodology)
            else:
                narrative = TemplateEngine().generate(results, meta, test_methodology=test_methodology) + "\n\n(Note: LLM weights were missing; output defaulted to template mode.)"
        else:
            engine = TemplateEngine()
            narrative = engine.generate(results, meta, test_methodology=test_methodology)

        # Build plot image for PDF (light background for printing)
        plot_path = REPORT_DIR / f"{session_id}_plot.png"
        build_shmoo_plot(df, results, save_path=str(plot_path))

        # Render PDF
        report_path = REPORT_DIR / f"{session_id}_report.pdf"
        generator = ReportGenerator()
        generator.generate(
            results=results,
            meta=meta,
            narrative=narrative,
            plot_path=str(plot_path),
            output_path=str(report_path),
            test_methodology=test_methodology
        )

        return send_file(
            str(report_path),
            as_attachment=True,
            download_name=f"{test_methodology}_SHMOO_Analysis_Report_{meta.get('die_id','D0001')}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


def _serialize_results(results) -> dict:
    return {
        'accuracy': round(results.accuracy, 4),
        'cv_accuracy': round(results.cv_accuracy, 4),
        'cv_std': round(results.cv_std, 4),
        'boundary_slope': round(results.boundary_slope, 4),
        'boundary_intercept': round(results.boundary_intercept, 4),
        'boundary_r2': round(results.boundary_r2, 4),
        'recommended_vdd': round(results.recommended_vdd, 3),
        'recommended_freq': round(results.recommended_freq, 3),
        'voltage_margin_v': round(results.voltage_margin_v, 3),
        'freq_margin_ghz': round(results.freq_margin_ghz, 3),
        'n_pass': results.n_pass,
        'n_fail': results.n_fail,
        'failure_code_dist': results.failure_code_dist,
        'critical_fault_patterns': getattr(results, 'critical_fault_patterns', results.timing_fail_patterns),
        'timing_fail_patterns': results.timing_fail_patterns,
        'is_multi_die': getattr(results, 'is_multi_die', False),
        'die_rankings': getattr(results, 'die_rankings', []),
        'high_performer': getattr(results, 'high_performer', None),
        'low_performer': getattr(results, 'low_performer', None),
        'die_results': getattr(results, 'die_results', {}),
    }


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
