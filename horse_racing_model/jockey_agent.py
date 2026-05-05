import asyncio
from pathlib import Path
from google import genai
from dotenv import load_dotenv
import json
import time
import pandas as pd

start_time = time.time()
load_dotenv()
client = genai.Client()

with open('jockey_prompt.txt', 'r') as f:
    system_prompt = f.read()

pathlist = list(Path("jockey_info").glob("*.pdf"))
print(f"Found {len(pathlist)} PDFs")

semaphore = asyncio.Semaphore(50)
save_lock = asyncio.Lock()
completed = []
output_file = 'jockey-4.xlsx'

async def process_jockey(path):
    async with semaphore:
        print(f"  Processing: {path.name}")
        try:
            with open(path, 'rb') as f:
                pdf_data = f.read()
        except FileNotFoundError:
            print(f"  -> File disappeared before reading, skipping.")
            return None

        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-3-flash-preview",
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0
                    ),
                    contents=[
                        genai.types.Part.from_bytes(
                            data=pdf_data,
                            mime_type="application/pdf"
                        )
                    ]
                )

                tokens = response.usage_metadata.total_token_count
                print(f"  -> {path.name}: {tokens} tokens")

                jockey = json.loads(response.text)
                if isinstance(jockey, list):
                    jockey = jockey[0]
                jockey['name'] = path.stem
                jockey['_tokens'] = tokens

                async with save_lock:
                    completed.append(jockey)
                    if len(completed) % 10 == 0:
                        save_df = pd.DataFrame(completed)
                        save_df.drop(columns=['_tokens'], errors='ignore').to_excel(output_file, index=False)
                        print(f"  [SAVED] Progress: {len(completed)} jockeys saved")

                return jockey

            except json.JSONDecodeError as e:
                print(f"  -> JSON parse error for {path.name}: {e}")
                print(f"  -> Raw: {response.text[:200]}")
                return None
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  -> Rate limited on {path.name}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  -> Error for {path.name}: {e}")
                    return None

async def main():
    tasks = [process_jockey(path) for path in pathlist]
    await asyncio.gather(*tasks)

    total_tokens = sum(h.pop('_tokens', 0) for h in completed)

    print(f"\nTotal tokens used: {total_tokens}")
    print(f"Successful: {len(completed)} / {len(pathlist)} jockeys")

    df = pd.DataFrame(completed)
    df.to_excel(output_file, index=False)
    print(f"Final save to {output_file}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

asyncio.run(main())