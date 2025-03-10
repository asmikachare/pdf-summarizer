import PyPDF2
from transformers import pipeline
import argparse

def extracttextfrompdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    # Open the file in binary mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        # Iterate over each page in the PDF
        for page in reader.pages:
            # Extract text from the page
            pagetext = page.extract_text()
            if pagetext:
                # Append extracted text to overall text variable (with a space to separate pages)
                text += pagetext + " "
    return text

def chunk_text(text, chunk_size=100):
    """
    Split the text into smaller chunks of 'chunk_size' words each.
    This helps avoid token limit issues when summarizing long text.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def remove_repeated_lines(summary_text):
    """
    Remove exact duplicate lines and normalize spacing.
    This helps to clean up the final summary output.
    """
    lines = summary_text.split('\n')
    new_lines = []
    seen = set()
    for line in lines:
        normalized_line = ' '.join(line.split())
        if normalized_line and normalized_line not in seen:
            new_lines.append(normalized_line)
            seen.add(normalized_line)
    return "\n".join(new_lines)

def format_summary(summary_text):
    """
    Format the summary text by adding a dash at the start of each line
    to create a bullet-point effect.
    """
    lines = summary_text.split('\n')
    formatted_lines = ["- " + line.strip() for line in lines if line.strip()]
    return "\n".join(formatted_lines)

def summarizetext(text, max_length=100, min_length=50):
    """
    Summarize the provided text using a summarization-specific model.
    Here we use the Bart-based summarizer (sshleifer/distilbart-cnn-12-6).
    The text is processed in chunks to avoid token limit issues.
    """
    # Initialize the summarization pipeline with a dedicated model
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaries = []
    # Process the text in chunks
    for chunk in chunk_text(text, chunk_size=100):
        # Summarize each chunk
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    # Combine all chunk summaries
    return "\n\n".join(summaries)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="PDF Summarizer CLI")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    print("Extracting text from PDF...")
    # Extract text from the provided PDF file
    text = extracttextfrompdf(args.pdf_path)
    if not text:
        print("No text could be extracted from the PDF.")
        return

    print("Summarizing text...")
    # Generate a raw summary from the extracted text
    raw_summary = summarizetext(text)
    # Remove repeated lines to clean up the output
    cleaned_summary = remove_repeated_lines(raw_summary)
    # Format the summary to include bullet points
    formatted_summary = format_summary(cleaned_summary)

    print("\n--- Summary ---")
    print(formatted_summary)

if __name__ == "__main__":
    main()
