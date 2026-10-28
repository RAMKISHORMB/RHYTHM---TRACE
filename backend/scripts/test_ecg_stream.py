
import asyncio
import json

from websockets.asyncio.client import connect


async def main():
    uri = "ws://127.0.0.1:8000/ws/ecg"

    async with connect(uri) as websocket:
        for index in range(3):
            frame = json.loads(await websocket.recv())

            if frame.get("type") != "ecg_frame":
                raise ValueError(f"Unexpected response: {frame}")

            assert frame["fs"] == 250
            assert frame["simulated"] is False
            assert frame["replayed"] is True
            assert len(frame["ecg_raw"]) == 25

            print(f"\nFRAME {index + 1}")
            print("Record:", frame["record_id"])
            print("Source:", frame["source"])
            print("First sample:", frame["t0_sample"])
            print("Sampling rate:", frame["fs"])
            print("Samples received:", len(frame["ecg_raw"]))
            print("First five ECG values:", frame["ecg_raw"][:5])
            print("Annotations:", frame["beats"])

    print("\nSUCCESS: Real ECG replay WebSocket verified.")


if __name__ == "__main__":
    asyncio.run(main())
