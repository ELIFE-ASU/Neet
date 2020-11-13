.. currentmodule:: neet

.. testsetup:: introduction

      from neet import Information
      from neet.boolean.examples import s_pombe, s_cerevisiae

.. _information:

Information Analysis
====================

Out of the box, Neet provides facilities for computing a few common information-theoretic quantities
from networks. All of these methods rely on constructing time series, from which a collection of
probabilities distributions are built. The :class:`Information` class provides a simple mechanism
for automating this process, and caching results for relatively efficient computation.

Initialization
--------------

Constructing an instance of :class:`Information`, you simply provide a network, a history length
(used to compute measures such as `active information
<api/information.html#neet.Information.active_information>`__ or `transfer entropy
<api/information.html#neet.Information.transfer_entropy>`__), and the length of time series to
compute.

.. doctest:: introduction

      >>> Information(s_pombe, k=5, timesteps=20)
      <neet.information.Information object at 0x...>

At initialization, a `time series <api/landscape.html#neet.Landscape.timeseries>`__ is computed
based on the parameters provided. This is cached and used whenever you request an information
measure.

Of course, you can override the parameters after initialization, and the time series will be
recomputed.

.. doctest:: introduction

      >>> arch = Information(s_pombe, k=5, timesteps=20)
      >>> arch.net = s_cerevisiae
      >>> arch.k = 2
      >>> arch.timesteps = 100

Information Measures
--------------------

Once you have an :class:`Information` instance, you can request an informormation measure. This will
compute and cache the value.

.. doctest:: introduction

      >>> arch = Information(s_pombe, k=5, timesteps=20)
      >>> arch.active_information()  # computed and cached
      array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
             0.40046165, 0.67019615, 0.67019615, 0.39189127])
      >>> arch.active_information()  # cached value is returned
      array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
             0.40046165, 0.67019615, 0.67019615, 0.39189127])

Each information measure is only computed and cached when you request it. In the event that you
change some aspect of the information architecture, e.g. the network, the cache of information
measures is also cleared.

.. doctest:: introduction

      >>> arch = Information(s_pombe, k=5, timesteps=20)
      >>> arch.active_information()
      array([0.        , 0.4083436 , 0.62956679, 0.62956679, 0.37915718,
             0.40046165, 0.67019615, 0.67019615, 0.39189127])
      >>> arch.net = s_cerevisiae
      >>> arch.active_information()
      array([0.        , 0.35677758, 0.410884  , 0.44191249, 0.54392362,
             0.42523414, 0.35820287, 0.13355861, 0.42823889, 0.22613507,
             0.28059538])
