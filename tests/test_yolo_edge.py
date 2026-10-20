from pathlib import Path

from yolo_edge import SRTBox, parse_srt_boxes, simulate_box_data_for_frame


def test_parse_srt_boxes_from_label_payload(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "box: 40,30,180,120\n\n"
        "2\n"
        "00:00:02,500 --> 00:00:04,000\n"
        "110, 80, 250, 170\n",
        encoding="utf-8",
    )

    boxes = parse_srt_boxes(srt_path)
    assert len(boxes) == 2
    assert boxes[0] == SRTBox(0.0, 2.0, 40.0, 30.0, 180.0, 120.0)
    assert boxes[1].x1 == 110.0
    assert boxes[1].y1 == 80.0


def test_simulate_box_data_for_frame() -> None:
    boxes = [
        SRTBox(start_seconds=0.0, end_seconds=1.0, x1=50.0, y1=60.0, x2=150.0, y2=180.0),
    ]

    simulated = simulate_box_data_for_frame((200, 300, 3), boxes, timestamp_seconds=0.5, frame_index=5)

    assert len(simulated) == 1
    assert simulated[0]["confidence"] > 0.0
    assert len(simulated[0]["xyxy"]) == 4
    assert simulated[0]["xyxy"][0] >= 0.0
    assert simulated[0]["xyxy"][0] <= 299.0
    assert simulated[0]["xyxy"][1] >= 0.0
    assert simulated[0]["xyxy"][1] <= 199.0
    assert simulated[0]["xyxy"][2] > simulated[0]["xyxy"][0]
    assert simulated[0]["xyxy"][3] > simulated[0]["xyxy"][1]
    assert simulated[0]["xyxy"][2] <= 299.0
    assert simulated[0]["xyxy"][3] <= 199.0
