
import asyncio
import json
import sys

import websockets


URL = "ws://127.0.0.1:8000/ws/artifact-demo"

EXPECTED_WINDOW_TRUST = [
    "HIGH",
    "REVIEW",
    "REVIEW",
    "UNRELIABLE",
    "UNRELIABLE",
]


async def main():
    prediction_count = 0
    frame_count = 0
    total_beats = 0
    completed = False

    async with websockets.connect(
        URL,
        max_size=2**22,
    ) as websocket:

        async for raw_message in websocket:
            data = json.loads(raw_message)
            message_type = data.get("type")

            if message_type == "artifact_prediction":
                index = data["window_index"]
                trust = data["trust"]
                beats = data["beats"]

                assert index == prediction_count
                assert trust["level"] == EXPECTED_WINDOW_TRUST[index]
                assert data["beat_count"] == len(beats)

                for beat in beats:
                    assert (
                        beat["absolute_sample"]
                        == data["window_start_sample"]
                        + beat["relative_sample"]
                    )

                    assert beat["trust_level"] in (
                        "HIGH",
                        "REVIEW",
                        "UNRELIABLE",
                    )

                    assert (
                        beat["annotation_source"]
                        == "dataset_annotation"
                    )

                    # A local beat assessment must not
                    # upgrade its parent window trust.
                    rank = {
                        "UNRELIABLE": 0,
                        "REVIEW": 1,
                        "HIGH": 2,
                    }

                    assert (
                        rank[beat["trust_level"]]
                        <= rank[trust["level"]]
                    )

                prediction_count += 1
                total_beats += len(beats)

                print(
                    f"Window {index + 1}: "
                    f"window_trust={trust['level']} | "
                    f"aligned_beats={len(beats)}"
                )

            elif message_type == "artifact_ecg_frame":
                frame_count += 1

            elif message_type == "artifact_demo_complete":
                completed = True
                break

            elif message_type == "error":
                raise RuntimeError(data["message"])

    assert completed
    assert prediction_count == 5
    assert frame_count == 200
    assert total_beats == 25

    print()
    print("Predictions:", prediction_count)
    print("ECG frames:", frame_count)
    print("Aligned beat assessments:", total_beats)
    print("TRACE BEAT WEBSOCKET TEST PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("TEST FAILED:", exc)
        sys.exit(1)
