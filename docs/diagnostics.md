# Diagnostic collection

Console and aggregate timing collection are disabled by default. Terminal and
file logging levels control log output independently of these collectors.

Pass these switches before the subcommand:

- `--tokio-console` starts Tokio Console on `127.0.0.1:6677` in the main
  process and `127.0.0.1:6667` in the IPC worker.
- `--aggregate-timing` collects busy and elapsed times for spans declaring
  `agg_perf` and prints the aggregate table at shutdown. It enables those spans
  even with INFO log output, while unrelated DEBUG spans stay filtered unless
  another layer needs them.

The main process forwards both switches to its IPC worker, including replacement
workers. Enable either switch independently or combine them. The existing
`--write-perf-log-file` switch controls a separate performance output file and
does not enable either collector.
