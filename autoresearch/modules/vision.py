"""Vision Module - Extracts figures, charts, and visual content from academic PDFs."""
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class VisionModule:
    """
    Computer vision capabilities for academic document analysis.
    Extracts figures, charts, tables, and generates visual summaries.
    """

    def __init__(self):
        self.has_opencv = self._check_opencv()
        self.has_pil = self._check_pil()

    def _check_opencv(self) -> bool:
        try:
            import cv2
            return True
        except ImportError:
            return False

    def _check_pil(self) -> bool:
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def extract_figures_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract figure metadata and images from PDF."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            figures = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract images
                images = page.get_images(full=True)
                for img_idx, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"]

                    figures.append({
                        "page": page_num + 1,
                        "index": img_idx,
                        "format": ext,
                        "size_bytes": len(image_bytes),
                        "type": "image",
                        "caption": self._find_caption(page, img_idx)
                    })

                # Detect figure captions
                text = page.get_text()
                caption_pattern = r'(?:Figure|Fig\.?)\s*(\d+)[\.:]?\s*(.*?)(?=\n|$)'
                for match in re.finditer(caption_pattern, text, re.IGNORECASE):
                    figures.append({
                        "page": page_num + 1,
                        "figure_number": match.group(1),
                        "caption": match.group(2).strip(),
                        "type": "caption"
                    })

            doc.close()
            return figures

        except Exception as e:
            return [{"error": str(e), "type": "extraction_failed"}]

    def _find_caption(self, page, img_index: int) -> str:
        """Find caption near an image."""
        text = page.get_text()
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if f"Figure {img_index + 1}" in line or f"Fig. {img_index + 1}" in line:
                return line.strip()
        return ""

    def detect_chart_type(self, image_path: str) -> Dict[str, Any]:
        """Detect if an image is a chart/graph and classify its type."""
        if not self.has_opencv:
            return {"error": "OpenCV not available"}

        try:
            import cv2
            import numpy as np

            img = cv2.imread(image_path)
            if img is None:
                return {"error": "Could not load image"}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Count lines (high line count suggests chart)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=30, maxLineGap=10)
            line_count = len(lines) if lines is not None else 0

            # Detect circles (pie charts, scatter plots)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                      param1=50, param2=30, minRadius=5, maxRadius=50)
            circle_count = len(circles[0]) if circles is not None else 0

            # Classify
            if line_count > 50 and circle_count < 5:
                chart_type = "line_chart_or_bar_chart"
            elif circle_count > 3:
                chart_type = "pie_chart_or_scatter"
            elif line_count > 20:
                chart_type = "table_or_grid"
            else:
                chart_type = "general_image"

            return {
                "chart_type": chart_type,
                "line_count": line_count,
                "circle_count": circle_count,
                "is_chart": line_count > 20 or circle_count > 3
            }

        except Exception as e:
            return {"error": str(e)}

    def generate_citation_network_viz(self, network_data: Dict) -> Dict[str, Any]:
        """Generate visualization data for citation network."""
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])

        # Create D3.js compatible format
        viz_nodes = []
        for i, node in enumerate(nodes):
            viz_nodes.append({
                "id": i,
                "label": node.get("title", "")[:30],
                "group": node.get("domains", ["unknown"])[0] if node.get("domains") else "unknown",
                "size": max(5, min(30, (node.get("citations", 0) or 0) / 50 + 5)),
                "year": node.get("year")
            })

        viz_links = []
        for edge in edges:
            viz_links.append({
                "source": edge["source"],
                "target": edge["target"],
                "value": edge.get("weight", 0.5)
            })

        return {
            "nodes": viz_nodes,
            "links": viz_links,
            "format": "d3_force_directed",
            "num_nodes": len(viz_nodes),
            "num_links": len(viz_links)
        }

    def generate_trend_chart(self, year_distribution: Dict[str, int]) -> Dict[str, Any]:
        """Generate time-series chart data for publication trends."""
        years = sorted([int(y) for y in year_distribution.keys() if y.isdigit()])
        counts = [year_distribution[str(y)] for y in years]

        return {
            "chart_type": "line_chart",
            "x_axis": "Year",
            "y_axis": "Publications",
            "data": [{"year": y, "count": c} for y, c in zip(years, counts)],
            "trend": "increasing" if len(counts) > 1 and counts[-1] > counts[0] else "stable"
        }
