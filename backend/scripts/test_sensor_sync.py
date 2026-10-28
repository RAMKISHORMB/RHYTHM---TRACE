
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.sensor_sync import synchronize_sensors


def expect_error(**kwargs):
    try:
        synchronize_sensors(**kwargs)
    except ValueError:
        return

    raise AssertionError("Expected ValueError.")


def main():
    ecg = np.arange(1000, dtype=float)
    imu = np.sin(np.arange(1000) / 25)

    result = synchronize_sensors(
        ecg=ecg,
        imu=imu,
        ecg_fs=250,
        imu_fs=250,
    )

    assert result.ecg.shape == (1000,)
    assert result.imu.shape == (1000, 1)
    assert result.imu_channels == 1
    assert np.allclose(result.imu[:, 0], imu)
    print("TEST 1 PASSED: 250 Hz alignment")

    # 100 Hz IMU covering the same 4-second interval
    # with an extra endpoint sample for interpolation.
    imu_100 = np.arange(401) / 100

    result = synchronize_sensors(
        ecg=ecg,
        imu=imu_100,
        ecg_fs=250,
        imu_fs=100,
    )

    assert np.allclose(
        result.imu[:, 0],
        result.timestamps,
    )
    print("TEST 2 PASSED: 100 Hz interpolation")

    imu_3axis = np.column_stack([
        imu,
        imu * 2,
        imu * 3,
    ])

    result = synchronize_sensors(
        ecg=ecg,
        imu=imu_3axis,
        ecg_fs=250,
        imu_fs=250,
    )

    assert result.imu.shape == (1000, 3)
    print("TEST 3 PASSED: Three-axis IMU")

    expect_error(
        ecg=ecg,
        imu=imu[:100],
        ecg_fs=250,
        imu_fs=250,
    )
    print("TEST 4 PASSED: Incomplete IMU rejected")

    expect_error(
        ecg=ecg,
        imu=imu,
        ecg_fs=0,
        imu_fs=250,
    )
    print("TEST 5 PASSED: Invalid rate rejected")

    expect_error(
        ecg=ecg,
        imu=np.full(1000, np.nan),
        ecg_fs=250,
        imu_fs=250,
    )
    print("TEST 6 PASSED: Invalid IMU values rejected")

    print()
    print("ALL 6 SENSOR SYNCHRONIZATION TESTS PASSED")


if __name__ == "__main__":
    main()
