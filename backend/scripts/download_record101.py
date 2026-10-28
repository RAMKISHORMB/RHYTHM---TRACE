
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "mitdb101"
DEST.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://physionet.org/files/mitdb/1.0.0"
FILES = ["101.hea", "101.dat", "101.atr"]

MAX_RETRIES = 5
CHUNK_SIZE = 16 * 1024


def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    target = DEST / filename
    partial = DEST / f"{filename}.part"

    print(f"\nDownloading {filename}...", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        offset = partial.stat().st_size if partial.exists() else 0

        headers = {}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(15, 20),
            ) as response:

                response.raise_for_status()

                # A server may ignore a Range request.
                # Never append a full response to partial data.
                if offset and response.status_code == 200:
                    print(
                        "Server did not resume; restarting file.",
                        flush=True,
                    )
                    offset = 0

                if offset and response.status_code != 206:
                    raise RuntimeError(
                        "Unexpected response to resume request."
                    )

                if offset and not response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-"):
                    raise RuntimeError(
                        "Server returned an unexpected byte range."
                    )

                length = response.headers.get("Content-Length")
                expected = (
                    offset + int(length)
                    if length is not None
                    else None
                )

                mode = "ab" if offset else "wb"

                with open(partial, mode) as file:
                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if chunk:
                            file.write(chunk)

                downloaded = partial.stat().st_size

                if expected is None:
                    raise RuntimeError(
                        "Cannot verify download size."
                    )

                if downloaded != expected:
                    raise RuntimeError(
                        f"Incomplete download: "
                        f"{downloaded}/{expected} bytes"
                    )

                partial.replace(target)

                print(
                    f"Saved {filename}: "
                    f"{downloaded:,} bytes",
                    flush=True,
                )
                return

        except (
            requests.RequestException,
            RuntimeError,
        ) as error:
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed: "
                f"{error}",
                flush=True,
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(2)


def main():
    print("TRACE — RESUMABLE RECORD 101 DOWNLOAD")
    print("------------------------------------")

    for filename in FILES:
        download_file(filename)

    print("\nALL THREE FILES DOWNLOADED SUCCESSFULLY")


if __name__ == "__main__":
    main()
