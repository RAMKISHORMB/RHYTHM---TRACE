
import asyncio
import json

import websockets


URL = "ws://127.0.0.1:8000/ws/artifact-demo"

EXPECTED_WINDOWS = 5
EXPECTED_FRAMES_PER_WINDOW = 40


async def main():
    predictions = {}
    audits = {}
    frame_counts = {}
    completed = False

    print("TRACE — LIVE AUDIT WEBSOCKET TEST")
    print("Connecting to:", URL)

    async with websockets.connect(
        URL,
        open_timeout=15,
        max_size=10_000_000,
    ) as websocket:

        while True:
            raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=30,
            )

            message = json.loads(raw)
            message_type = message.get("type")

            if message_type == "error":
                raise RuntimeError(
                    message.get("message", "Unknown server error")
                )

            if message_type == "artifact_prediction":
                index = message["window_index"]

                assert index not in predictions
                predictions[index] = message

                print(
                    f"Window {index}: "
                    f"prediction={message['predicted_artifact']}, "
                    f"trust={message['trust']['level']}"
                )

            elif message_type == "artifact_audit":
                index = message["window_index"]

                assert index not in audits
                assert index in predictions

                audit = message["audit"]

                assert audit["window_index"] == index
                assert audit["event_id"]
                assert audit["timestamp_utc"]
                assert audit["explanation"]
                assert audit["clinical_validated"] is False

                audits[index] = audit

                print(
                    f"Window {index}: "
                    f"audit={audit['action']}"
                )

            elif message_type == "artifact_ecg_frame":
                index = message["window_index"]

                assert index in predictions
                assert index in audits

                frame_counts[index] = (
                    frame_counts.get(index, 0) + 1
                )

                assert len(
                    message["ecg_reference"]
                ) == len(
                    message["ecg_corrupted"]
                )

            elif message_type == "artifact_demo_complete":
                assert (
                    message["windows_streamed"]
                    == EXPECTED_WINDOWS
                )
                completed = True
                break

    assert completed

    assert len(predictions) == EXPECTED_WINDOWS
    assert len(audits) == EXPECTED_WINDOWS

    for index in range(EXPECTED_WINDOWS):
        assert frame_counts.get(index) == (
            EXPECTED_FRAMES_PER_WINDOW
        )

        prediction = predictions[index]
        audit = audits[index]

        trust_level = prediction["trust"]["level"]

        assert audit["trust_level"] == trust_level
        assert audit["artifact_class"] == (
            prediction["predicted_artifact"]
        )

        if trust_level == "UNRELIABLE":
            assert audit["action"] == "ABSTAINED"
        else:
            assert audit["action"] == "PRESERVED"

    print()
    print("TEST 1 PASSED: 5 prediction events")
    print("TEST 2 PASSED: 5 audit events")
    print("TEST 3 PASSED: 200 ECG frames")
    print("TEST 4 PASSED: Audit-window alignment")
    print("TEST 5 PASSED: Conservative trust policy")
    print()
    print("ALL 5 LIVE WEBSOCKET TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
