# SPDX-FileCopyrightText: © 2024-2026 Jimmy Fitzpatrick <jimmy@spectregrams.org>
# This file is part of SPECTRE
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses

from gnuradio import spectre
from gnuradio import soapy

import spectre_server.core.fields

from ._base import Base, BaseModel
from ._constants import FlowgraphConstant


@dataclasses.dataclass(frozen=True)
class RX888MK2Port:
    """Specifies one of the antenna ports on the RX-888 MK II."""

    HF = "HF"
    VHF = "VHF"


class RX888MK2FixedCenterFrequencyModel(BaseModel):
    sample_rate: spectre_server.core.fields.Field.sample_rate = 8e6
    center_frequency: spectre_server.core.fields.Field.center_frequency = 32e6
    antenna_port: spectre_server.core.fields.Field.antenna_port = RX888MK2Port.HF
    if_gain: spectre_server.core.fields.Field.if_gain = 20
    rf_gain: spectre_server.core.fields.Field.rf_gain = 0
    bias_tee: spectre_server.core.fields.Field.bias_tee = False
    batch_size: spectre_server.core.fields.Field.batch_size = 1
    buffers: spectre_server.core.fields.Field.buffers = 1024
    output_type: spectre_server.core.fields.Field.output_type = (
        spectre_server.core.fields.OutputType.FC32
    )


class RX888MK2FixedCenterFrequency(Base[RX888MK2FixedCenterFrequencyModel]):
    def configure(self, tag: str, model: RX888MK2FixedCenterFrequencyModel) -> None:
        device = "driver=SDDC"
        type = model.output_type
        nchan = 1
        dev_args = ""
        stream_args = f"buffers={model.buffers}"
        tune_args = [""]
        settings = [""]
        self.soapy_rx888mk2_source = soapy.source(
            device, type, nchan, dev_args, stream_args, tune_args, settings
        )
        self.soapy_rx888mk2_source.set_sample_rate(0, model.sample_rate)
        self.soapy_rx888mk2_source.set_frequency(0, model.center_frequency)
        self.soapy_rx888mk2_source.set_antenna(0, model.antenna_port)
        self.soapy_rx888mk2_source.set_gain(0, "RF", model.rf_gain)
        self.soapy_rx888mk2_source.set_gain(0, "IF", model.if_gain)
        self.soapy_rx888mk2_source.write_setting(
            "UpdBiasT_HF", "true" if model.bias_tee else "false"
        )

        self.spectre_batched_file_sink = spectre.batched_file_sink(
            self._batches_dir_path,
            tag,
            model.output_type,
            model.batch_size,
            model.sample_rate,
            FlowgraphConstant.GROUP_BY_DATE,
        )
        self.connect(
            (self.soapy_rx888mk2_source, 0), (self.spectre_batched_file_sink, 0)
        )
