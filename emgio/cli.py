"""Command-line interface for emgio / biosigIO.

A THIN wrapper over the public API: ``convert`` (import -> EDF/BDF export),
``verify`` (compare two recordings), and ``info`` (summarize a recording). It
contains no signal math or format-decision logic; those live in the core.

Diagnostics (including the chatty per-importer/exporter ``print`` output) go to
stderr; a ``--json`` payload is written alone to stdout so it can be piped.

Exit-code contract:
    0  success (verify passed)
    1  verify FAILED (signals differ beyond tolerance)
    2  usage / argument error (bad args, unknown --modality, unsupported format)
    3  input file not found or unreadable
    4  conversion runtime error
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from collections.abc import Sequence

from . import __version__
from .analysis.verification import compare_signals
from .bids import find_events_tsv
from .core.emg import EMG
from .core.modality import VALID_MODALITIES, infer_modality_from_channel_type

logger = logging.getLogger("emgio.cli")

# Exit codes (see module docstring).
EXIT_OK = 0
EXIT_VERIFY_FAILED = 1
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_RUNTIME = 4

# A channel is "unknown" when its type is the importer's unrecognized default
# (``OTHER``): --modality may fill it, and info/verify warn-list it. Modality is
# NOT a usable marker here -- ECG/EOG/ACC/TRIG all legitimately resolve to MISC.
_UNKNOWN_CHANNEL_TYPES = frozenset({"OTHER", "", None})


class CliError(Exception):
    """A CLI failure carrying the exit code to return."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


@contextlib.contextmanager
def _quiet_stdout():
    """Route library ``print`` chatter to stderr so stdout stays clean."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _load_emg(path: str) -> EMG:
    """Load a recording, mapping failures onto the exit-code contract."""
    if not os.path.exists(path):
        raise CliError(EXIT_INPUT, f"input not found: {path}")
    if not os.path.isfile(path):
        raise CliError(EXIT_INPUT, f"input is not a file: {path}")
    try:
        with _quiet_stdout():
            return EMG.from_file(path)
    except ValueError as e:
        if "unsupported" in str(e).lower():
            raise CliError(EXIT_USAGE, str(e)) from e
        raise CliError(EXIT_RUNTIME, f"failed to read {path}: {e}") from e
    except CliError:
        raise
    except Exception as e:  # importer/runtime failure
        raise CliError(EXIT_RUNTIME, f"failed to read {path}: {e}") from e


def _is_unknown(info: dict) -> bool:
    return info.get("channel_type", "OTHER") in _UNKNOWN_CHANNEL_TYPES


def _unknown_channels(emg: EMG) -> list[str]:
    return [name for name, info in emg.channels.items() if _is_unknown(info)]


def _apply_modality(emg: EMG, modality: str) -> None:
    """Fill the modality of UNKNOWN channels only; never override detected ones.

    Source-detected modalities win; omitting --modality (this is not called)
    never injects anything. An unknown --modality is a usage error.
    """
    canonical = modality.strip().upper()
    if canonical not in VALID_MODALITIES:
        raise CliError(
            EXIT_USAGE,
            f"unknown --modality '{modality}'; choose one of {sorted(VALID_MODALITIES)}",
        )
    for info in emg.channels.values():
        if _is_unknown(info):
            info["modality"] = canonical


def _parse_channel_map(spec: str | None) -> dict[str, str] | None:
    """Parse ``A=B,C=D`` into a channel map, or None."""
    if not spec:
        return None
    mapping: dict[str, str] = {}
    for pair in spec.split(","):
        if "=" not in pair:
            raise CliError(EXIT_USAGE, f"invalid --channel-map entry '{pair}'; expected A=B")
        original, reloaded = pair.split("=", 1)
        mapping[original.strip()] = reloaded.strip()
    return mapping


def _all_identical(results: dict) -> bool:
    compared = [v for k, v in results.items() if k != "channel_summary"]
    return bool(compared) and all(v["is_identical"] for v in compared)


def _written_path(requested: str) -> str:
    """The path the exporter actually wrote (auto format may switch .edf/.bdf)."""
    if os.path.exists(requested):
        return requested
    base = os.path.splitext(requested)[0]
    for ext in (".bdf", ".edf"):
        if os.path.exists(base + ext):
            return base + ext
    return requested


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_convert(args: argparse.Namespace) -> int:
    emg = _load_emg(args.input)
    if args.modality:
        _apply_modality(emg, args.modality)

    unknown = _unknown_channels(emg)
    if unknown:
        logger.warning("channels with unknown modality: %s", ", ".join(unknown))

    try:
        with _quiet_stdout():
            emg.to_edf(
                args.output,
                format=args.format,
                create_channels_tsv=not args.no_channels_tsv,
                bypass_analysis=args.bypass_analysis,
            )
    except CliError:
        raise
    except Exception as e:
        raise CliError(EXIT_RUNTIME, f"conversion failed: {e}") from e

    written = _written_path(args.output)
    if not os.path.exists(written):
        raise CliError(EXIT_RUNTIME, f"conversion produced no output for {args.output}")
    logger.info("wrote %s", written)

    if args.verify:
        try:
            with _quiet_stdout():
                reloaded = EMG.from_file(written)
            results = compare_signals(emg, reloaded, tolerance=args.verify_tolerance)
        except Exception as e:
            raise CliError(EXIT_RUNTIME, f"verify reload failed: {e}") from e
        if not _all_identical(results):
            differing = [
                k for k, v in results.items() if k != "channel_summary" and not v["is_identical"]
            ]
            logger.error("verify FAILED; channels differ: %s", ", ".join(differing))
            return EXIT_VERIFY_FAILED
        logger.info("verify passed: all channels identical within tolerance")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    original = _load_emg(args.original)
    reloaded = _load_emg(args.reloaded)
    channel_map = _parse_channel_map(args.channel_map)
    try:
        results = compare_signals(
            original, reloaded, tolerance=args.tolerance, channel_map=channel_map
        )
    except ValueError as e:  # e.g. channel-map names not present
        raise CliError(EXIT_USAGE, str(e)) from e

    passed = _all_identical(results)
    if args.json:
        payload = {
            "passed": passed,
            "tolerance": args.tolerance,
            "channels": {
                k: {
                    "nrmse": v["nrmse"],
                    "max_norm_abs_diff": v["max_norm_abs_diff"],
                    "is_identical": v["is_identical"],
                }
                for k, v in results.items()
                if k != "channel_summary"
            },
            "summary": results.get("channel_summary", {}),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        for name, v in results.items():
            if name == "channel_summary":
                continue
            status = "ok" if v["is_identical"] else "DIFF"
            print(
                f"  [{status}] {name}: nrmse={v['nrmse']:.3e} maxdiff={v['max_norm_abs_diff']:.3e}"
            )
        print(f"verify {'passed' if passed else 'FAILED'} (tolerance {args.tolerance})")
    return EXIT_OK if passed else EXIT_VERIFY_FAILED


def cmd_info(args: argparse.Namespace) -> int:
    emg = _load_emg(args.input)
    rates = sorted({float(i["sample_frequency"]) for i in emg.channels.values()})
    n_samples = int(emg.signals.shape[0]) if emg.signals is not None else 0
    max_rate = max(rates) if rates else 0.0
    duration = round(n_samples / max_rate, 6) if max_rate else 0.0
    n_events = int(len(emg.events)) if emg.events is not None else 0
    events_tsv = find_events_tsv(args.input)
    unknown = _unknown_channels(emg)

    channels = [
        {
            "name": name,
            "type": info.get("channel_type"),
            "modality": info.get("modality")
            or infer_modality_from_channel_type(info.get("channel_type", "OTHER")),
            "sample_frequency": float(info["sample_frequency"]),
            "unit": info.get("physical_dimension"),
        }
        for name, info in emg.channels.items()
    ]

    if unknown:
        logger.warning("channels with unknown modality: %s", ", ".join(unknown))

    if args.json:
        payload = {
            "file": args.input,
            "n_channels": len(channels),
            "sampling_rates": rates,
            "duration_s": duration,
            "n_events": n_events,
            "events_tsv": events_tsv,
            "unknown_channels": unknown,
            "channels": channels,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"file: {args.input}")
        print(f"channels: {len(channels)} | sampling rates: {rates} Hz | duration: {duration}s")
        print(f"events: {n_events} | sibling events.tsv: {events_tsv or 'none'}")
        if unknown:
            print(f"unknown-modality channels: {', '.join(unknown)}")
        for ch in channels:
            print(
                f"  {ch['name']}: type={ch['type']} modality={ch['modality']} "
                f"fs={ch['sample_frequency']}Hz unit={ch['unit']}"
            )
    return EXIT_OK


def cmd_lowres(args: argparse.Namespace) -> int:
    """Down-sample a recording and export a lightweight low-res EDF/BDF.

    Thin wrapper: the anti-aliased resampling lives in ``EMG.resample`` and the
    scaling/format bracketing in the EDF exporter. ``--bits 16`` -> EDF (16-bit),
    ``--bits 24`` -> BDF (24-bit). Default is "double low-res": 16-bit + 100 Hz.
    """
    emg = _load_emg(args.input)
    if args.modality:
        _apply_modality(emg, args.modality)

    rates = sorted({float(i["sample_frequency"]) for i in emg.channels.values()})
    source_rate = max(rates) if rates else 0.0

    # Skip the resample step when the source is already at or below the target
    # rate; up-sampling is out of scope (and EMG.resample would refuse it).
    if source_rate <= args.rate:
        print(
            f"note: source rate {source_rate} Hz already <= target {args.rate} Hz; "
            "exporting without resampling",
            file=sys.stderr,
        )
    else:
        try:
            emg = emg.resample(args.rate)
        except ValueError as e:
            raise CliError(EXIT_RUNTIME, f"resample failed: {e}") from e

    fmt = "edf" if args.bits == 16 else "bdf"
    try:
        with _quiet_stdout():
            emg.to_edf(
                args.output,
                format=fmt,
                create_channels_tsv=not args.no_channels_tsv,
                bypass_analysis=True,
            )
    except CliError:
        raise
    except Exception as e:
        raise CliError(EXIT_RUNTIME, f"lowres export failed: {e}") from e

    written = _written_path(args.output)
    if not os.path.exists(written):
        raise CliError(EXIT_RUNTIME, f"lowres produced no output for {args.output}")
    logger.info("wrote %s", written)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biosigio",
        description="Convert, verify, and inspect biosignal recordings (EMG/EEG/iEEG/MEG).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase verbosity (repeatable)"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="import a recording and export EDF/BDF")
    p_convert.add_argument("input", help="input recording (any supported format)")
    p_convert.add_argument("output", help="output .edf/.bdf path")
    p_convert.add_argument("--format", choices=["auto", "edf", "bdf"], default="auto")
    p_convert.add_argument("--modality", help="fill UNKNOWN channels' modality (never overrides)")
    p_convert.add_argument(
        "--no-channels-tsv", action="store_true", help="do not write a BIDS channels.tsv"
    )
    p_convert.add_argument("--verify", action="store_true", help="reload and compare after export")
    p_convert.add_argument("--verify-tolerance", type=float, default=1e-4)
    p_convert.add_argument(
        "--bypass-analysis",
        action="store_true",
        help="skip signal analysis (requires --format edf|bdf)",
    )
    p_convert.set_defaults(func=cmd_convert)

    p_verify = sub.add_parser("verify", help="compare two recordings channel-by-channel")
    p_verify.add_argument("original", help="reference recording")
    p_verify.add_argument("reloaded", help="recording to compare against the reference")
    p_verify.add_argument("--tolerance", type=float, default=1e-4)
    p_verify.add_argument("--channel-map", help="map original->reloaded names, e.g. A=B,C=D")
    p_verify.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p_verify.set_defaults(func=cmd_verify)

    p_info = sub.add_parser("info", help="summarize a recording's channels and metadata")
    p_info.add_argument("input", help="input recording (any supported format)")
    p_info.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p_info.set_defaults(func=cmd_info)

    p_lowres = sub.add_parser(
        "lowres", help="down-sample (anti-aliased) and export a lightweight EDF/BDF"
    )
    p_lowres.add_argument("input", help="input recording (any supported format)")
    p_lowres.add_argument("output", help="output .edf/.bdf path")
    p_lowres.add_argument(
        "--rate", type=float, default=100.0, help="target sample rate in Hz (default: 100)"
    )
    p_lowres.add_argument(
        "--bits",
        type=int,
        choices=[16, 24],
        default=16,
        help="output bit depth: 16 -> EDF, 24 -> BDF (default: 16)",
    )
    p_lowres.add_argument("--modality", help="fill UNKNOWN channels' modality (never overrides)")
    p_lowres.add_argument(
        "--no-channels-tsv", action="store_true", help="do not write a BIDS channels.tsv"
    )
    p_lowres.set_defaults(func=cmd_lowres)

    return parser


def _configure_logging(verbose: int, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    # force=True overrides any root config already installed at import time
    # (e.g. emgio.core.emg calls basicConfig on import), so -v/-q actually apply
    # and CLI diagnostics reach stderr.
    logging.basicConfig(
        level=level, stream=sys.stderr, format="%(levelname)s: %(message)s", force=True
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code (does not raise SystemExit)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # argparse exits 2 on bad args / 0 on --help/--version
        return int(e.code) if e.code is not None else EXIT_USAGE

    _configure_logging(args.verbose, args.quiet)
    try:
        return args.func(args)
    except CliError as e:
        logger.error("%s", e)
        return e.code
    except KeyboardInterrupt:  # pragma: no cover
        return EXIT_RUNTIME


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
