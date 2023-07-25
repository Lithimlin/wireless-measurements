#!/usr/bin/env python3

import logging
import os
import signal
import time
from multiprocessing import Process
from pprint import pprint
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wifi_info import wireless
from wifi_info.settings import InfluxDBSettings, InterfaceSettings, Iperf3Settings
from wifi_info.storage.influxdb import InfluxDBStorage
from wifi_info.utils import InterfaceError


class CollectionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    iperf3: Iperf3Settings
    influxdb: InfluxDBSettings
    wireless_interface: Optional[InterfaceSettings] = Field(None, required=False)
    loop_iperf: bool = Field(False, required=False)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def int_handler(signum, frame):
    logging.info(f"Received {signal.strsignal(signum)} signal")
    logging.info("Stopping execution loop...")
    global LOOP_EXECUTION
    LOOP_EXECUTION = False


def main():
    global LOOP_EXECUTION
    LOOP_EXECUTION = True

    setup_logging()
    settings = CollectionSettings()

    db = InfluxDBStorage(settings.influxdb)

    def write_iperf3_metrics():
        iperf3_metrics = wireless.get_iperf3_metrics(settings.iperf3)
        db.write_data(iperf3_metrics)

    signal.signal(signal.SIGINT, int_handler)

    logging.info(
        f"Running iperf3 {'until interrupted' if settings.loop_iperf else 'once'}"
    )
    while LOOP_EXECUTION:
        LOOP_EXECUTION = settings.loop_iperf
        proc = Process(target=write_iperf3_metrics)
        proc.start()

        while proc.is_alive():
            if settings.wireless_interface is None:
                continue

            try:
                wireless_metrics = wireless.get_wireless_metrics(
                    settings.wireless_interface
                )
                db.write_data(wireless_metrics)
            except InterfaceError as e:
                logging.error(e)

            time.sleep(1)

    logging.info("Done!")


if __name__ == "__main__":
    main()
