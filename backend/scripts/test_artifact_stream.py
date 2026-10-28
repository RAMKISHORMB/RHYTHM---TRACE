
import asyncio
import json

import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws/artifact-demo"

    predictions = []
    frame_count = 0

    print("Connecting to TRACE artefact WebSocket...")

    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "artifact_prediction":
                predictions.append(data)

                print(
                    f"Window {data['window_index'] + 1}: "
                    f"expected={data['true_artifact']} | "
                    f"predicted={data['predicted_artifact']} | "
                    f"probability={data['model_probability']:.2%}"
                )

            elif message_type == "artifact_ecg_frame":
                frame_count += 1

            elif message_type == "artifact_demo_complete":
                print("\nDemo completed.")
                print("Predictions received:", len(predictions))
                print("ECG frames received:", frame_count)
                break

            elif message_type == "error":
                raise RuntimeError(data.get("message", "Unknown error"))

    assert len(predictions) == 5, "Expected five predictions."
    assert frame_count == 200, "Expected 200 ECG frames."

    print("\nTRACE ARTEFACT WEBSOCKET TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
