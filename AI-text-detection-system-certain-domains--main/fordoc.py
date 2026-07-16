import pdfplumber
import pytesseract
from ai_detector_pipeline import detect_ai, confidence_score

# Fixed document path requested by user
PDF_FILE_PATH = r"C:\Users\kbhan\Desktop\testpdf.pdf"    

def extract_pages(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            # OCR fallback
            if not text or len(text.strip()) < 50:
                image = page.to_image(resolution=300).original
                text = pytesseract.image_to_string(image)

            pages.append({
                "page": i + 1,
                "text": text.strip() if text else ""
            })

    return pages

def analyze_document(testpdf=PDF_FILE_PATH):
    pages = extract_pages(testpdf)
    results = []

    for page in pages:
        text = page["text"]

        if len(text.split()) < 30:
            continue  # skip weak pages

        detection = detect_ai(text)

        final = detection.get("final_score", detection.get("final_ai_probability", 0.0))

# 🔥 ADD THIS LINE HERE
        conf = confidence_score(detection, text)

        results.append({
            "page": page["page"],
            "ai_score": detection.get("classifier_score", 0.0),
            "perplexity": detection.get("perplexity", 0.0),
            "stylometry": detection.get("stylometry_score", 0.0),
            "final_score": final,
            "confidence": conf,            # 🔥 ADD THIS
            "word_count": len(text.split())
            })

    return results

def compute_overall_ai(results):
    total_words = 0
    weighted_score = 0

    for r in results:
        word_count = r["word_count"]

        total_words += word_count
        weighted_score += r["final_score"] * word_count

    return (weighted_score / total_words) * 100 if total_words > 0 else 0

def find_max_ai_page(results):
    if not results:
        return None

    return max(results, key=lambda x: x["final_score"])

def get_top_ai_pages(results, top_n=2):
    if not results:
        return []

    sorted_pages = sorted(results, key=lambda x: x["final_score"], reverse=True)
    return sorted_pages[:top_n]

def generate_final_report(testpdf=PDF_FILE_PATH):
    results = analyze_document(testpdf)

    overall_ai = compute_overall_ai(results)
    top_pages = get_top_ai_pages(results, top_n=2)

    print("\n===== FINAL REPORT =====\n")

    print(f"Overall AI Likelihood: {overall_ai:.2f}%\n")

    if top_pages:
        print("Pages with Highest AI Likelihood:\n")
        for p in top_pages:
            print(f"Page {p['page']} → {p['final_score']*100:.2f}%")

if __name__ == '__main__':
    # always use fixed path as requested
    generate_final_report(PDF_FILE_PATH)
