# Data Directory

This directory is for **sample input files** during development and testing.

## Purpose

The data folder serves as a convenient location to store test files locally while developing the application. However, these files are:

- **NOT required** for the application to run
- **NOT committed** to version control (gitignored)
- **Only for local testing** purposes

## Supported File Types

The Streamlit application accepts uploads for:

### Documents
- **PDF files** (`.pdf`) - Research papers, articles, documents
- **Text files** (`.txt`) - Plain text content

### Audio
- **Audio files** (`.wav`, `.mp3`) - For transcription and analysis

### Web Content
- **URLs** - Paste URLs directly into the app for web scraping

## Usage

### For Local Development

1. Place your test files in this directory:
   ```
   data/
   ├── sample_paper.pdf
   ├── research_notes.txt
   └── interview.wav
   ```

2. These files are gitignored and won't be committed to the repository

3. The application will use the Streamlit file uploader instead of hardcoded paths

### For Production (Streamlit Cloud)

In production, users will upload files directly through the Streamlit interface. No files from this directory are deployed or used in production.

## Why is this directory empty?

To keep the repository lightweight and fast to clone, we don't commit large media files to version control. Instead:

- Users upload their own files through the web interface
- Sample files are generated on-the-fly during testing
- The directory structure is preserved but without large binary files

## Adding Your Own Test Files

Feel free to add your own test files to this directory for local development:

```bash
# Example: Add a research paper
cp ~/Downloads/my_paper.pdf data/

# Example: Add test audio
cp ~/Downloads/interview.wav data/
```

Remember: These files will **not** be committed thanks to `.gitignore` rules.
