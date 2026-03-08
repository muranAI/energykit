energykit Documentation
=======================

**The Python toolkit that turns energy data into dollars.**

Most energy tools stop at the metric. energykit goes all the way to the money.

.. code-block:: python

   import energykit as ek

   report = ek.diagnose(your_meter_data)
   print(report.total_addressable_savings_usd)   # e.g. 1453.21

.. grid:: 2

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/diagnose
   api/cost
   api/anomaly
   api/forecast
   api/optimize
   api/features
   api/benchmark
   api/datasets

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub <https://github.com/muranAI/energykit>
   PyPI <https://pypi.org/project/energykit/>
   Muranai.com <https://muranai.com>
