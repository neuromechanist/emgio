# Command-Line Interface

Installing biosigIO provides a `biosigio` console command (entry point
`biosigio.cli:main`). Verify it with `biosigio --help`.

Subcommands:

- `biosigio convert IN OUT` — import a recording and export EDF/BDF.
- `biosigio verify A B` — compare two recordings channel-by-channel.
- `biosigio info IN` — summarize a recording's channels and metadata.
- `biosigio lowres IN OUT` — write a downsampled, low-resolution copy.

Run `biosigio <subcommand> --help` for the full options of each.

## Module Documentation

::: biosigio.cli
    options:
      show_root_heading: true
      show_source: true
      members: true
