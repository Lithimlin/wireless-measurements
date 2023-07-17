#!/usr/bin/env python3

import logging
import os
import time
from multiprocessing import Process
from pprint import pprint

from wifi_info import wireless
from wifi_info.settings import Settings
from wifi_info.storage.influxdb import InfluxDBStorage
from wifi_info.utils import InterfaceError


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()
    settings = Settings()

    db = InfluxDBStorage(settings.influxdb)

    def write_iperf3_metrics():
        iperf3_metrics = wireless.get_iperf3_metrics(settings.iperf3)
        db.write_data(iperf3_metrics)

    proc = Process(target=write_iperf3_metrics)
    proc.start()

    while proc.is_alive():
        try:
            wireless_metrics = wireless.get_wireless_metrics(
                settings.wireless_interface
            )
            db.write_data(wireless_metrics)
        except InterfaceError as e:
            logging.error(e)
        time.sleep(1)


if __name__ == "__main__":
    main()
