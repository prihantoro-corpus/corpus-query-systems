import argparse
import sys
import os
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.preprocessing.corpus_loader import load_monolingual_corpus_files

def main():
    parser = argparse.ArgumentParser(description="Convert Text/XML corpora to a pre-compiled DuckDB database for CORTEX.")
    parser.add_argument("input_files", nargs="+", help="Path to input text or XML files")
    parser.add_argument("-o", "--output", default="compiled_corpus.db", help="Output database filename (e.g. corpus.db)")
    parser.add_argument("-l", "--lang", default="English", help="Language of the corpus (e.g. English, Indonesian)")
    parser.add_argument("-f", "--format", default="Raw (Natural text)", help="Format: 'Raw (Natural text)' or 'Tagged (Vertical)'")

    args = parser.parse_args()

    file_sources = []
    for filepath in args.input_files:
        if not os.path.exists(filepath):
            print(f"Error: File not found - {filepath}")
            sys.exit(1)
        
        # Open in binary mode as expected by the loader
        f = open(filepath, 'rb')
        file_sources.append(f)

    def print_progress(val, text):
        print(f"[{int(val*100)}%] {text}")

    print(f"Processing {len(file_sources)} files...")
    
    result = load_monolingual_corpus_files(
        file_sources, 
        explicit_lang_code=args.lang, 
        selected_format=args.format, 
        progress_callback=print_progress
    )

    # Close all files
    for f in file_sources:
        f.close()

    if result.get('error'):
        print(f"\nError during processing: {result['error']}")
        sys.exit(1)
    else:
        temp_db = result['db_path']
        shutil.copy(temp_db, args.output)
        print(f"\nSuccessfully created database: {args.output}")
        print(f"   Total Tokens: {result['stats'].get('total_tokens', 0)}")
        print(f"   You can now upload '{args.output}' to the online version of CORTEX!")

if __name__ == "__main__":
    main()
