"""
Автотест сценария «Верификация поступающих данных от симулятора».

Допущение о протоколе (в исходном задании он не задан):
    TCP, один JSON-объект на строку (разделитель newline).

    {
      "signals": [
        {"id": 1, "name": "sin_1", "value": 0.12, "quality": "Good", "time": "2026-10-14T08:00:00.000Z"},
        ...
      ]
    }

    Ровно 10 сигналов, поля как в таблице главного окна: id, name, value, quality, time.

Боевой симулятор слушает порт 2001. Здесь берём эфемерный порт (bind(..., 0)),
чтобы тест не зависел от занятого 2001 и запускался без реальных приложений.

Почему этого достаточно для случайных значений:
    Вручную тестировщик не угадает ГСЧ и не обязан проверять формулу синуса.
    Автотест не предсказывает value — он запоминает кадр в момент отправки
    (эталон) и сверяет его с тем, что получилось после приёма и разбора
    (таблица). Расхождение эталона и таблицы = ложная телеметрия.
    Проверки вида assert len(table) > 0 недостаточно: какие-то цифры пришли
    не доказывает, что система работает корректно.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest

SIGNAL_COUNT = 10
QUALITY_GOOD = "Good"


@dataclass(frozen=True)
class SignalRow:
    """Одна строка таблицы: то, что оператор должен увидеть в UI."""

    id: int
    name: str
    value: float
    quality: str
    time: str


def make_frame(seq: int) -> list[SignalRow]:
    """Собирает эталонный кадр из 10 сигналов.

    seq отличает кадры друг от друга: те же id/name, другие value и time.
    value намеренно «случайные» с точки зрения проверяющего — в тесте мы
    не восстанавливаем закон изменения, а сравниваем два конца канала.
    """
    base = datetime(2026, 10, 14, 8, 0, 0, tzinfo=timezone.utc)
    stamp = (base + timedelta(seconds=seq)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    # Детерминированный псевдослучайный ряд: кадр воспроизводим, но не
    # проверяем «это синус». Эталон — сам список, а не формула.
    values = [((seq * 17 + i * 13) % 100) / 10.0 for i in range(SIGNAL_COUNT)]
    return [
        SignalRow(
            id=i + 1,
            name=f"signal_{i + 1}",
            value=values[i],
            quality=QUALITY_GOOD,
            time=stamp,
        )
        for i in range(SIGNAL_COUNT)
    ]


def frame_to_json_line(rows: list[SignalRow]) -> bytes:
    payload = {
        "signals": [
            {
                "id": row.id,
                "name": row.name,
                "value": row.value,
                "quality": row.quality,
                "time": row.time,
            }
            for row in rows
        ]
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def parse_table(raw_line: bytes) -> list[SignalRow]:
    """Модель клиента: байты с сокета -> строки таблицы.

    В реальном приложении здесь же жила бы отрисовка UI. Для верификации
    данных важен именно этот слой: то, что ушло, должно совпасть с тем,
    что получилось после десериализации.
    """
    payload = json.loads(raw_line.decode("utf-8"))
    return [
        SignalRow(
            id=int(item["id"]),
            name=str(item["name"]),
            value=float(item["value"]),
            quality=str(item["quality"]),
            time=str(item["time"]),
        )
        for item in payload["signals"]
    ]


class FakeSimulator:
    """TCP-симулятор: по подключению отдаёт заранее сохранённые эталонные кадры."""

    def __init__(self, frames: list[list[SignalRow]]) -> None:
        self.frames = frames
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Порт 0 = эфемерный. Боевой сервис из задания — 2001.
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self._sock.settimeout(2)
        self.host, self.port = self._sock.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._stopped = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        try:
            # Будим accept(), если он ещё ждёт клиента.
            with socket.create_connection((self.host, self.port), timeout=0.2):
                pass
        except OSError:
            pass
        self._sock.close()
        self._thread.join(timeout=2)

    def _serve(self) -> None:
        try:
            client, _addr = self._sock.accept()
        except (TimeoutError, OSError):
            return
        with client:
            for frame in self.frames:
                if self._stopped.is_set():
                    return
                client.sendall(frame_to_json_line(frame))


@pytest.fixture
def two_frames() -> list[list[SignalRow]]:
    # Два кадра: неизменные id/name, новые value и time — как живая подписка.
    return [make_frame(seq=1), make_frame(seq=2)]


@pytest.fixture
def simulator(two_frames: list[list[SignalRow]]) -> Iterator[FakeSimulator]:
    sim = FakeSimulator(two_frames)
    sim.start()
    try:
        yield sim
    finally:
        sim.stop()


def read_table_frames(host: str, port: int, count: int) -> list[list[SignalRow]]:
    """Главное окно в упрощении: подключились, прочитали count кадров в таблицу."""
    with socket.create_connection((host, port), timeout=2) as conn:
        conn_file = conn.makefile("rb")
        frames = []
        for _ in range(count):
            line = conn_file.readline()
            assert line, "симулятор закрыл соединение раньше, чем отдал ожидаемые кадры"
            frames.append(parse_table(line))
        return frames


def test_displayed_data_matches_what_simulator_sent(
    simulator: FakeSimulator,
    two_frames: list[list[SignalRow]],
) -> None:
    """Сценарий: данные в таблице — это ровно то, что отправил симулятор."""
    displayed = read_table_frames(simulator.host, simulator.port, count=2)

    assert len(displayed) == 2, "должны прийти оба кадра подписки"

    for frame_index, (sent, table) in enumerate(zip(two_frames, displayed)):
        # Недостаточно «что-то пришло»: кадр должен совпасть с эталоном полями.
        assert len(table) == SIGNAL_COUNT, (
            f"кадр {frame_index}: в таблице {len(table)} строк, ожидалось {SIGNAL_COUNT}"
        )
        assert [row.id for row in table] == [row.id for row in sent]
        assert [row.name for row in table] == [row.name for row in sent]
        # Это и есть ответ на «случайные значения не проверить вручную»:
        # сравниваем value с тем, что ушло, а не с формулой синуса.
        assert [row.value for row in table] == [row.value for row in sent]
        assert [row.quality for row in table] == [row.quality for row in sent]
        assert [row.time for row in table] == [row.time for row in sent]
        # Полное равенство строк — страховка, что не потеряли поле.
        assert table == sent

        for row in table:
            assert row.quality, f"кадр {frame_index}, id={row.id}: качество пустое"

    first, second = displayed
    assert [row.id for row in first] == [row.id for row in second]
    assert [row.name for row in first] == [row.name for row in second]
    assert [row.value for row in first] != [row.value for row in second], (
        "значение должно меняться между кадрами, иначе поток выглядит застывшим"
    )
    assert [row.time for row in first] != [row.time for row in second], (
        "метка времени должна меняться, иначе оператор не увидит, что кадр новый"
    )
