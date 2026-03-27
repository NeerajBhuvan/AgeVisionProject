"""
HTML Report Generator
======================
Generates a visual HTML report with side-by-side comparisons,
metrics tables, and test results.
"""

import base64
import logging
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("age_pipeline.report")


def _img_to_base64(img_bgr: np.ndarray, max_size: int = 300) -> str:
    """Convert BGR image to base64-encoded JPEG for HTML embedding."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _img_path_to_base64(path: str, max_size: int = 300) -> str:
    """Load image from path and convert to base64."""
    img = cv2.imread(path)
    if img is None:
        return ""
    return _img_to_base64(img, max_size)


def generate_report(results: list, output_path: str, title: str = "Age Progression Report"):
    """Generate an HTML report with all progression results.

    Args:
        results: List of result dicts, each containing:
            - input_path: Path to input image
            - input_name: Filename of input image
            - original_crop: BGR numpy array of original face crop
            - progressions: Dict of {age: {"image": bgr_array, "metrics": metrics_dict}}
            - grid_path: Path to comparison grid image (optional)
            - test_result: "PASS" or "FAIL" with reason
        output_path: Path to save the HTML report.
        title: Report title.
    """
    html_parts = [_HTML_HEADER.format(title=title, date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]

    # Summary table
    total = len(results)
    passed = sum(1 for r in results if r.get("test_result", "").startswith("PASS"))
    failed = total - passed

    html_parts.append(f"""
    <div class="summary">
        <h2>Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card pass">
                <div class="stat-value">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card fail">
                <div class="stat-value">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
        </div>
    </div>
    """)

    # Per-image results
    for idx, result in enumerate(results):
        input_name = result.get("input_name", f"Image {idx + 1}")
        test_result = result.get("test_result", "N/A")
        test_class = "pass" if test_result.startswith("PASS") else "fail"

        html_parts.append(f"""
        <div class="image-result">
            <h3>{input_name}
                <span class="badge {test_class}">{test_result.split(':')[0]}</span>
            </h3>
        """)

        # Original image
        if "original_crop" in result and result["original_crop"] is not None:
            orig_b64 = _img_to_base64(result["original_crop"])
            html_parts.append(f"""
            <div class="comparisons">
                <div class="img-card">
                    <img src="data:image/jpeg;base64,{orig_b64}" alt="Original">
                    <div class="img-label">Original</div>
                </div>
            """)

            # Aged images
            progressions = result.get("progressions", {})
            for age, prog_data in sorted(progressions.items()):
                if "image" in prog_data and prog_data["image"] is not None:
                    aged_b64 = _img_to_base64(prog_data["image"])
                    html_parts.append(f"""
                <div class="img-card">
                    <img src="data:image/jpeg;base64,{aged_b64}" alt="Age {age}">
                    <div class="img-label">Age {age}</div>
                </div>
                    """)

            html_parts.append("</div>")  # Close comparisons

        # Metrics table
        progressions = result.get("progressions", {})
        if any("metrics" in p for p in progressions.values()):
            html_parts.append("""
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Target Age</th>
                        <th>SSIM</th>
                        <th>PSNR (dB)</th>
                        <th>Identity Score</th>
                        <th>Est. Age</th>
                        <th>Age Error</th>
                        <th>Age Accuracy</th>
                    </tr>
                </thead>
                <tbody>
            """)
            for age, prog_data in sorted(progressions.items()):
                m = prog_data.get("metrics", {})
                if m:
                    id_class = "good" if m.get("identity_score", 0) > 0.7 else "warn"
                    html_parts.append(f"""
                    <tr>
                        <td>{age}</td>
                        <td>{m.get('ssim', 'N/A')}</td>
                        <td>{m.get('psnr', 'N/A')}</td>
                        <td class="{id_class}">{m.get('identity_score', 'N/A')}</td>
                        <td>{m.get('estimated_age', 'N/A')}</td>
                        <td>{m.get('age_error', 'N/A')}</td>
                        <td>{m.get('age_accuracy', 'N/A')}%</td>
                    </tr>
                    """)
            html_parts.append("</tbody></table>")

        # Test details
        if ":" in test_result:
            reason = test_result.split(":", 1)[1].strip()
            html_parts.append(f'<div class="test-detail">{reason}</div>')

        html_parts.append("</div>")  # Close image-result

    # Comparison grid images
    html_parts.append("<h2>Comparison Grids</h2>")
    for result in results:
        grid_path = result.get("grid_path")
        if grid_path and Path(grid_path).exists():
            grid_b64 = _img_path_to_base64(grid_path, max_size=1200)
            if grid_b64:
                html_parts.append(f"""
                <div class="grid-section">
                    <h4>{result.get('input_name', 'Unknown')}</h4>
                    <img src="data:image/jpeg;base64,{grid_b64}" alt="Grid"
                         class="grid-img">
                </div>
                """)

    html_parts.append(_HTML_FOOTER)

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    logger.info("Report saved to %s", output_path)


# ─── HTML Template ───────────────────────────────────────────

_HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', -apple-system, sans-serif;
        background: #0f1117;
        color: #e1e4e8;
        padding: 2rem;
        line-height: 1.6;
    }}
    h1 {{
        text-align: center;
        font-size: 2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #7F5AF0, #2CB67D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .date {{ text-align: center; color: #72757e; margin-bottom: 2rem; }}
    h2 {{ color: #7F5AF0; margin: 2rem 0 1rem; border-bottom: 1px solid #2d2f36; padding-bottom: 0.5rem; }}
    h3 {{ color: #e1e4e8; margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem; }}

    .summary {{ background: #1a1d27; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
    .stats-grid {{ display: flex; gap: 1rem; justify-content: center; }}
    .stat-card {{
        background: #242732;
        border-radius: 8px;
        padding: 1.5rem 2.5rem;
        text-align: center;
        min-width: 120px;
    }}
    .stat-card.pass {{ border-left: 4px solid #2CB67D; }}
    .stat-card.fail {{ border-left: 4px solid #E16259; }}
    .stat-value {{ font-size: 2rem; font-weight: bold; }}
    .stat-label {{ color: #72757e; font-size: 0.85rem; margin-top: 0.25rem; }}

    .badge {{
        font-size: 0.75rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .badge.pass {{ background: rgba(44,182,125,0.2); color: #2CB67D; }}
    .badge.fail {{ background: rgba(225,98,89,0.2); color: #E16259; }}

    .image-result {{
        background: #1a1d27;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .comparisons {{
        display: flex;
        gap: 1rem;
        overflow-x: auto;
        padding: 1rem 0;
    }}
    .img-card {{
        flex: 0 0 auto;
        text-align: center;
    }}
    .img-card img {{
        width: 200px;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid #2d2f36;
    }}
    .img-label {{
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #a0a4ab;
    }}

    .metrics-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
    }}
    .metrics-table th {{
        background: #242732;
        padding: 0.75rem;
        text-align: left;
        color: #7F5AF0;
        font-weight: 600;
    }}
    .metrics-table td {{
        padding: 0.75rem;
        border-bottom: 1px solid #2d2f36;
    }}
    .metrics-table .good {{ color: #2CB67D; font-weight: 600; }}
    .metrics-table .warn {{ color: #F0A500; font-weight: 600; }}

    .test-detail {{
        margin-top: 0.5rem;
        padding: 0.5rem 1rem;
        background: #242732;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #a0a4ab;
    }}

    .grid-section {{ margin: 1rem 0; }}
    .grid-img {{
        max-width: 100%;
        border-radius: 8px;
        border: 1px solid #2d2f36;
    }}

    footer {{
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2d2f36;
        color: #72757e;
        font-size: 0.8rem;
    }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="date">Generated: {date}</p>
"""

_HTML_FOOTER = """
<footer>
    <p>AgeVision Project &mdash; Age Progression Pipeline Report</p>
    <p>Powered by HRFAE (High Resolution Face Age Editing) GAN</p>
</footer>
</body>
</html>
"""
