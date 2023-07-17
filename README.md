# Wifi-Info

## Disclaimer

This is my first, half-hearted attempt at creating a python package just to get the hang of things.
I also used this opportunity to get into pydantic (v2.0) a bit.

Why is it on GitHub then? Because that made it easier to transfer between machines.

## Usage

Should you - for whatever reason - choose to use this "package", you'll have to install it yourself
as I did not publish it and do not intend to do so.

Then use the `template.env` file as a guide to create your `.env` file.

The `docker-compose.yml` lets you easily spin up an influxDB server on your local machine to store your metrics.

The main script is `collect-metrics.py` (at least for my purposes).
It will collect iperf3 metrics for a specified amount of time (default 30 seconds as found in the `template.env` file)
while monitoring the signal strength and other metrics from the specified wireless interface
and then exit.
