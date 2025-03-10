import PyPDF2
from transformers import pipeline
import argparse

def extracttextfrompdf(pdf_path):
    """Extract text from PDF using PyPDF2"""
    text = ""
    # Open the file in binary mode to read
    with open(pdf_path, 'rb') as file:
        # Create a reader object
        reader = PyPDF2.PdfReader(file)
        # Iterate through all pages in the PDF
        for page in reader.pages: 
            # Extract text from each page
            pagetext = page.extract_text()
            if pagetext:
                # Append page text to overall text variable
                text += pagetext
    return text

def summarizetext(text, max=150, min=40):
    """Summarize using Hugging Face's summarization pipeline"""
    # Load summarization pipeline (downloads default summarization model)
    summarizer = pipeline("summarization")
    # Generate summary using provided parameters
    summary = summarizer(text, max_length=max, min_length=min, do_sample=False)
    # Return summarized text
    return summary[0]['summary_text']

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="PDF Summarizer CLI")
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()

    print("Extracting text from PDF...")
    # Extract text from provided PDF file path
    text = extracttextfrompdf(args.pdf_path)

    if not text:
        print("No text could be extracted from the PDF.")
        return
    
    print("Summarizing text...")
    # Summarize extracted text
    summary = summarizetext(text)

    print("\n--- Summary ---")
    print(summary)

if __name__ == "__main__":
    main()
