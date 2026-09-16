# SPDX-FileCopyrightText: © 2024-2025 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import io
import typing

import typer
import requests
from PIL import Image

from ._secho_resources import secho_new_resource
from ._utils import safe_request, get_config_file_name, spinner

create_typer = typer.Typer(help="Create resources.")


@create_typer.command(help="Create a config.")
def config(
    receiver_name: str = typer.Option(
        ..., "--receiver", "-r", help="The name of the receiver."
    ),
    receiver_mode: str = typer.Option(
        ..., "--mode", "-m", help="The operating mode of the receiver."
    ),
    tag: str = typer.Option(None, "--tag", "-t", help="The unique identifier."),
    file_name: str = typer.Option(
        None, "-f", help="The file name.", metavar="<tag>.json"
    ),
    params: list[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Parameters as key-value pairs.",
        metavar="<key>=<value>",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="If specified, force the operation even if files exist with the same tag.",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="If specified, do not validate the parameters.",
    ),
) -> None:
    file_name = get_config_file_name(file_name, tag)

    json = {
        "receiver_name": receiver_name,
        "receiver_mode": receiver_mode,
        "string_parameters": params,
        "force": force,
        "validate": not skip_validation,
    }
    jsend_dict = safe_request(f"spectre-data/configs/{file_name}", "PUT", json=json)
    endpoint = jsend_dict["data"]
    secho_new_resource(endpoint)
    raise typer.Exit()


@create_typer.command(help="Copy an existing config under a new tag.")
def copy(
    tag: str = typer.Option(None, "--tag", "-t", help="The tag of the config to copy."),
    file_name: str = typer.Option(
        None, "-f", help="The file name of the config to copy.", metavar="<tag>.json"
    ),
    new_tag: str = typer.Option(None, "--new-tag", help="The tag for the new config."),
    new_file_name: str = typer.Option(
        None,
        "--new-file",
        help="The file name for the new config.",
        metavar="<tag>.json",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="If specified, force the operation even if files exist with the new tag.",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="If specified, do not validate the parameters.",
    ),
) -> None:
    src_file_name = get_config_file_name(file_name, tag)
    dst_file_name = get_config_file_name(new_file_name, new_tag)

    jsend_dict = safe_request(f"spectre-data/configs/{src_file_name}/raw", "GET")
    config = jsend_dict["data"]

    receiver_name = config["receiver_name"]
    receiver_mode = config["receiver_mode"]
    string_parameters = [f"{k}={v}" for k, v in config["parameters"].items()]

    json = {
        "receiver_name": receiver_name,
        "receiver_mode": receiver_mode,
        "string_parameters": string_parameters,
        "force": force,
        "validate": not skip_validation,
    }
    jsend_dict = safe_request(f"spectre-data/configs/{dst_file_name}", "PUT", json=json)
    endpoint = jsend_dict["data"]
    secho_new_resource(endpoint)
    raise typer.Exit()


@create_typer.command(help="Create a plot of spectrogram data.")
def plot(
    tags: list[str] = typer.Option(
        ...,
        "--tag",
        "-t",
        help="The file tag. Specifying multiple tags yields a stacked plot over the same time frame.",
    ),
    obs_date: typing.Optional[str] = typer.Option(
        None,
        "--obs-date",
        "--date",
        help="The start date of the observation, in the format `%Y-%m-%d`. Defaults to today.",
    ),
    start_time: str = typer.Option(
        "00:00:00",
        "--start-time",
        help="The start time of the observation (UTC), in the format `%H:%M:%S`.",
    ),
    end_time: str = typer.Option(
        "23:59:00",
        "--end-time",
        help="The end time of the observation (UTC), in the format `%H:%M:%S`.",
    ),
    lower_freq: float = typer.Option(
        None,
        "--lower-freq",
        help="The lower bound of the frequency range in Hz.",
    ),
    upper_freq: float = typer.Option(
        None,
        "--upper-freq",
        help="The upper bound of the frequency range in Hz.",
    ),
    log_norm: bool = typer.Option(
        False,
        "--log-norm",
        help="If specified, normalise all values to the 0-1 range on a logarithmic scale.",
    ),
    dBb: bool = typer.Option(
        False,
        "--db",
        "--dBb",
        help="If specified, use units of decibels above the background.",
    ),
    vmin: float = typer.Option(
        None,
        "--vmin",
        help="The minimum value for the colormap. Only applies if `dBb` is specified.",
    ),
    vmax: float = typer.Option(
        None,
        "--vmax",
        help="The maximum value for the colormap. Only applies if `dBb` is specified.",
    ),
    figsize_x: int = typer.Option(
        None, "--figsize-x", help="The horizontal size of the plot."
    ),
    figsize_y: int = typer.Option(
        None, "--figsize-y", help="The vertical size of the plot."
    ),
    mhz: bool = typer.Option(
        False,
        "--mhz",
        "--MHz",
        help="If specified, display frequencies in MHz instead of Hz.",
    ),
    elapsed_time: bool = typer.Option(
        False,
        "--elapsed-time",
        help="If specified, display elapsed seconds on the x-axis instead of UTC datetimes.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="If specified, download and open the plot after creation.",
    ),
) -> None:
    if obs_date is None:
        obs_date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    json = {
        "tags": tags,
        "obs_date": obs_date,
        "start_time": start_time,
        "end_time": end_time,
        "lower_freq": lower_freq,
        "upper_freq": upper_freq,
        "log_norm": log_norm,
        "dBb": dBb,
        "vmin": vmin,
        "vmax": vmax,
        "figsize_x": figsize_x,
        "figsize_y": figsize_y,
        "mhz": mhz,
        "elapsed_time": elapsed_time,
    }
    with spinner():
        jsend_dict = safe_request("spectre-data/batches/plots", "PUT", json=json)
    endpoint = jsend_dict["data"]
    secho_new_resource(endpoint)

    if show:
        response = requests.get(endpoint)
        response.raise_for_status()
        Image.open(io.BytesIO(response.content)).show()

    raise typer.Exit()
