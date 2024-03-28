# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations


def run():
    import streamlit.web.bootstrap

    from langsuite.webui.app import app_path

    streamlit.web.bootstrap.run(app_path, None, args=[], flag_options={})
