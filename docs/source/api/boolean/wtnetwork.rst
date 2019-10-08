Weight-Threshold Networks
-------------------------

.. automodule:: neet.boolean.wtnetwork
   :synopsis: Weight-Threshold Networks

.. autoclass:: neet.boolean.WTNetwork

   .. attribute:: weights

      The network's square weight matrix. The rows and columns are target and
      source nodes, respectively. That is, the :math:`(i,j)` element is the
      weight of the edge from the :math:`j`-th node to the :math:`i`-th.

      .. rubric:: Examples

      .. doctest:: wtnetwork

         >>> net = WTNetwork(3)
         >>> net.weights
         array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])

         >>> net = WTNetwork([[1, 0, 1], [-1, 1, 0], [0, 0, 1]])
         >>> net.weights
         array([[ 1.,  0.,  1.],
                [-1.,  1.,  0.],
                [ 0.,  0.,  1.]])

      :type: numpy.ndarray

   .. attribute:: thresholds

      The network's threshold vector. The :math:`i`-th element is the threshold
      for the :math:`i`-th node.

      .. rubric:: Examples

      .. doctest:: wtnetwork

         >>> net = WTNetwork(3)
         >>> net.thresholds
         array([0., 0., 0.])

         >>> net = WTNetwork(3, thresholds=[0, 0.5, -0.5])
         >>> net.thresholds
         array([ 0. ,  0.5, -0.5])

      :type: numpy.ndarray

   .. attribute:: theta

      The network's activation function. Every node in the network uses this
      function to determine its next state, based on the simulus it recieves.

      .. doctest:: wtnetwork

          >>> WTNetwork(3).theta
          <function WTNetwork.split_threshold at 0x...>
          >>> WTNetwork(3, theta=WTNetwork.negative_threshold).theta
          <function WTNetwork.negative_threshold at 0x...>

      This activation function must accept two arguments: the activation stimulus
      and the current state of the node or network. It should handle two types of
      arguments:

          1. stimulus and state are scalar
          2. stimulus and state are vectors (``list`` or :class:`numpy.ndarray`)

      In case 2, the result should `modify` the state in-place and return the vector.

      .. testcode:: wtnetwork

          def theta(stimulus, state):
              if isinstance(stimulus, (list, numpy.ndarray)):
                  for i, x in enumerate(stimulus):
                      state[i] = theta(x, state[i])
                  return state
              elif stimulus < 0:
                  return 0
              else:
                  return state
          net = WTNetwork(3, theta=theta)
          print(net.theta)

      .. testoutput:: wtnetwork

          <function theta at 0x...>

      As with all :class:`neet.Network` classes, the names of the nodes and
      network-wide metadata can be provided.

      :type: callable

   .. automethod:: read

   .. automethod:: positive_threshold

   .. automethod:: negative_threshold

   .. automethod:: split_threshold

