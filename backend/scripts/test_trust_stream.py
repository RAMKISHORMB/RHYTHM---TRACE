
import asyncio
import json
import sys

import websockets


URL = "ws://127.0.0.1:8000/ws/artifact-demo"

EXPECTED_LEVELS = [
    "HIGH",
    "REVIEW",
    "REVIEW",
    "UNRELIABLE",
    "UNRELIABLE",
]


async def main():
    predictions = []
    frames = 0
    completed = False

    async with websockets.connect(
        URL,
        max_size=2**22,
    ) as websocket:
        async for message in websocket:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "artifact_prediction":
                trust = data["trust"]

                predictions.append(data)

                print(
                    f"Window {data['window_index'] + 1}: "
                    f"predicted={data['predicted_artifact']} | "
                    f"trust={trust['level']} | "
                    f"score={trust['score']}"
                )

                assert trust["experimental"] is True
                assert trust["clinical_validated"] is False
                assert trust["reasons"]
                assert trust["policy_version"] == "trace-trust-v0.1"

            elif message_type == "artifact_ecg_frame":
                frames += 1

            elif message_type == "artifact_demo_complete":
                completed = True
                break

            elif message_type == "error":
                raise RuntimeError(data["message"])

    assert completed, "Demo did not complete."
    assert len(predictions) == 5, "Expected five predictions."
    assert frames == 200, f"Expected 200 frames, got {frames}."

    # These checks assume the current first five
    # demonstration windows remain in their known order.
    actual_levels = [
        item["trust"]["level"]
        for item in predictions
    ]

    assert actual_levels == EXPECTED_LEVELS, (
        f"Unexpected trust levels: {actual_levels}"
    )

    print("\nTRACE TRUST WEBSOCKET TEST PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("TEST FAILED:", exc)
        sys.exit(1)
