Installation
============

Requires Python 3.9 or higher.

Quick install
-------------

.. code-block:: bash

   pip install energykit

With optional extras
--------------------

.. code-block:: bash

   # + LightGBM forecasting (recommended for best accuracy)
   pip install "energykit[forecast]"

   # + Everything (forecast, DER optimizer, dataset downloaders)
   pip install "energykit[all]"

Install extras explained
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Extra
     - What it adds
     - When to use
   * - ``energykit[forecast]``
     - LightGBM, statsmodels
     - Better load forecasting accuracy
   * - ``energykit[optimize]``
     - PuLP solver
     - Advanced DER dispatch
   * - ``energykit[datasets]``
     - requests, tqdm
     - Auto-download public datasets
   * - ``energykit[all]``
     - Everything above
     - Development / full feature set

Windows
-------

.. code-block:: powershell

   python --version      # must be 3.9+
   pip install energykit
   python -c "import energykit; print(energykit.__version__)"

**Using Anaconda:**

.. code-block:: powershell

   conda create -n energykit-env python=3.11
   conda activate energykit-env
   pip install "energykit[all]"

macOS
-----

.. code-block:: bash

   pip3 install energykit
   python3 -c "import energykit; print(energykit.__version__)"

**Using Homebrew + pyenv:**

.. code-block:: bash

   brew install pyenv
   pyenv install 3.11 && pyenv global 3.11
   pip install "energykit[all]"

Linux
-----

**Ubuntu / Debian:**

.. code-block:: bash

   sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip -y
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install "energykit[all]"

**RHEL / Fedora:**

.. code-block:: bash

   sudo dnf install python3.11 python3-pip -y
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install "energykit[all]"
