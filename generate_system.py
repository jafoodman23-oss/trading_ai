import subprocess
import sys

# First, check if anthropic is installed. If not, install it.
try:
    import anthropic
except ImportError:
    print("[INFO] anthropic not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
    import anthropic

import os
import time
from pathlib import Path

# Configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY")
OUTPUT_DIR = Path("./generated_system")

if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable not set")
    print("\nTo set it, run this in PowerShell:")
    print('$env:ANTHROPIC_API_KEY = "your-api-key-here"')
    print("\nThen run this script again.")
    sys.exit(1)

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"[INFO] Output directory: {OUTPUT_DIR.absolute()}")

# Read the mega-prompt
prompt_file = Path("mega_prompt.txt")
if not prompt_file.exists():
    print(f"ERROR: {prompt_file} not found")
    print("Make sure mega_prompt.txt is in the same folder as this script")
    sys.exit(1)

with open(prompt_file, "r") as f:
    mega_prompt = f.read()

print(f"[INFO] Mega-prompt loaded ({len(mega_prompt):,} characters)")
print("[INFO] Starting autonomous code generation...")
print("[INFO] This will take a while. Do NOT interrupt.\n")

client = anthropic.Anthropic(api_key=API_KEY)

generated_code = ""
chunk_count = 0
retry_count = 0
total_tokens = 0
MAX_RETRIES = 10

def generate_chunk():
    global generated_code, chunk_count, retry_count, total_tokens
    
    if chunk_count == 0:
        full_prompt = mega_prompt
    else:
        full_prompt = mega_prompt + f"\n\n[CONTINUATION]\nYou have generated {len(generated_code):,} chars so far.\nLast 1000 chars:\n{generated_code[-1000:]}\n\nCONTINUE FROM HERE:"
    
    while retry_count < MAX_RETRIES:
        try:
            print(f"\n[GENERATION {chunk_count + 1}] Calling Claude API...")
            
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            response_text = message.content[0].text
            generated_code += response_text
            total_tokens += message.usage.input_tokens + message.usage.output_tokens
            chunk_count += 1
            retry_count = 0
            
            print(f"[SUCCESS] Generated {len(response_text):,} chars")
            print(f"[STATS] Total: {len(generated_code):,} chars, {total_tokens:,} tokens used")
            
            # Save progress
            with open(OUTPUT_DIR / "COMPLETE_SYSTEM.py", "w") as f:
                f.write(generated_code)
            
            return True
            
        except anthropic.RateLimitError:
            retry_count += 1
            wait_time = 65 * (2 ** (retry_count - 1))
            print(f"\n[RATE LIMIT] Hit limit. Retry {retry_count}/{MAX_RETRIES}")
            print(f"[WAITING] {wait_time} seconds...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                return False
            time.sleep(10)
    
    return False

# Generate
if not generate_chunk():
    print("[FAILED] Generation failed")
    sys.exit(1)

# Keep generating until system is complete
max_chunks = 5
while chunk_count < max_chunks:
    has_all = all(x in generated_code for x in ["def main", "class", "Backtest", "Strategy"])
    if has_all:
        print("\n[COMPLETE] System components detected")
        break
    
    print(f"\n[PHASE {chunk_count + 1}] Continuing generation...")
    if not generate_chunk():
        print("[WARNING] Generation stalled")
        break

# Save final results
with open(OUTPUT_DIR / "GENERATION_METADATA.txt", "w") as f:
    f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total characters: {len(generated_code):,}\n")
    f.write(f"Total tokens: {total_tokens:,}\n")
    f.write(f"Chunks: {chunk_count}\n")

print(f"\n[DONE] Generation complete!")
print(f"[OUTPUT] {OUTPUT_DIR / 'COMPLETE_SYSTEM.py'}")
print(f"[STATS] {len(generated_code):,} chars, {total_tokens:,} tokens, {chunk_count} chunks")
