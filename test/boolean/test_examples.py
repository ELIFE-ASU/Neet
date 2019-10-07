import unittest
import neet.boolean.examples as ex


def all_example_networks():
    return [
        ex.s_pombe,
        ex.s_cerevisiae,
        ex.c_elegans,
        ex.p53_no_dmg,
        ex.p53_dmg,
        ex.mouse_cortical_7B,
        ex.mouse_cortical_7C,
        ex.myeloid,
    ]


class TestBooleanExamples(unittest.TestCase):

    def test_examples_loaded(self):
        """
        Test that all example networks successfully load.
        """
        all_example_networks()

    def test_s_pombe(self):
        self.assertEqual(9, ex.s_pombe.size)
        self.assertEqual(["SK", "Cdc2_Cdc13", "Ste9", "Rum1", "Slp1",
                          "Cdc2_Cdc13_active", "Wee1_Mik1", "Cdc25",
                          "PP"], ex.s_pombe.names)

    def test_s_cerevisiae(self):
        self.assertEqual(11, ex.s_cerevisiae.size)
        self.assertEqual(["Cln3", "MBF", "SBF", "Cln1_2", "Cdh1", "Swi5",
                          "Cdc20_Cdc14", "Clb5_6", "Sic1", "Clb1_2",
                          "Mcm1_SFF"], ex.s_cerevisiae.names)

    def test_examples_metadata(self):
        """
        Test that all examples have name, description, and citation metadata.
        """
        for net in all_example_networks():
            self.assertTrue('name' in net.metadata)
            self.assertTrue('description' in net.metadata)
            self.assertTrue('citation' in net.metadata)

    def test_myeloid(self):
        self.assertEqual(11, ex.myeloid.size)
        self.assertEqual(['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1',
                          'SCL', 'C/EBPa', 'PU.1', 'cJun', 'EgrNab',
                          'Gfi-1'], ex.myeloid.names)
        self.assertEqual(ex.myeloid.update([1, 1, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0], 0),
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(ex.myeloid.update([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(ex.myeloid.update([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(ex.myeloid.update([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]),
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])

    def test_myeloid_attractors(self):
        # Assert attractors.
        attractors = ([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
        for state in attractors:
            self.assertEqual(ex.myeloid.update(state[:]), state)

        # Assert non-attractors.
        for state in ex.myeloid:
            if state not in attractors:
                self.assertNotEqual(ex.myeloid.update(state[:]), state)

    def test_myeloid_logic_expressions(self):
        self.assertEqual(['GATA-2', 'GATA-1', 'FOG-1', 'EKLF', 'Fli-1', 'SCL',
                          'C/EBPa', 'PU.1', 'cJun', 'EgrNab', 'Gfi-1'],
                         ex.myeloid_from_expr.names)
        self.assertEqual(ex.myeloid_from_expr.table, ex.myeloid.table)
        # Assert attractors.
        attractors = ([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
        for state in attractors:
            self.assertEqual(ex.myeloid_from_expr.update(state[:]), state)

        # Assert non-attractors.
        for state in ex.myeloid:
            if state not in attractors:
                self.assertNotEqual(
                    ex.myeloid_from_expr.update(state[:]), state)

    def test_mouse_cortical_7B(self):
        self.assertEqual(10, ex.mouse_cortical_7B.size)
        self.assertEqual(['gF', 'gE', 'gP', 'gC', 'gS', 'pF', 'pE',
                          'pP', 'pC', 'pS'], ex.mouse_cortical_7B.names)
        self.assertEqual(ex.mouse_cortical_7B.update([1, 1, 0, 0, 0, 0,
                                                      0, 0, 0, 0], 0),
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(ex.mouse_cortical_7B.update([1, 1, 0, 0, 0, 0,
                                                      0, 0, 0, 0]),
                         [0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(ex.mouse_cortical_7B.update([1, 1, 0, 0, 0, 0,
                                                      0, 0, 0, 0]),
                         [0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(ex.mouse_cortical_7B.update([0, 0, 0, 0, 0, 0,
                                                      0, 1, 1, 1]),
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_mouse_cortical_7B_attractors(self):
        # Assert attractors.
        attractors = (
            # Anterior desired attractor (reachable from desired anterior
            # initial state)
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],

            # Posterior desired attractor (unreachable from desired
            # posterior initial state)
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        )

        for state in attractors:
            self.assertEqual(ex.mouse_cortical_7B.update(state[:]), state)

        # Assert non-attractors.
        for state in ex.mouse_cortical_7B:
            if state not in attractors:
                self.assertNotEqual(
                    ex.mouse_cortical_7B.update(state[:]), state)

    def test_mouse_cortical_7B_logic_expressions(self):
        self.assertEqual(['gF', 'gE', 'gP', 'gC', 'gS', 'pF', 'pE',
                          'pP', 'pC', 'pS'],
                         ex.mouse_cortical_7B_from_expr.names)
        self.assertEqual(ex.mouse_cortical_7B_from_expr.table,
                         ex.mouse_cortical_7B.table)
        # Assert attractors.
        attractors = (
            # Anterior desired attractor (reachable from desired anterior
            # initial state)
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],

            # Posterior desired attractor (unreachable from desired
            # posterior initial state)
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        )

        for state in attractors:
            self.assertEqual(
                ex.mouse_cortical_7B_from_expr.update(state[:]), state)

        # Assert non-attractors.
        for state in ex.mouse_cortical_7B:
            if state not in attractors:
                self.assertNotEqual(
                    ex.mouse_cortical_7B_from_expr.update(state[:]), state)

    def test_mouse_cortical_7C(self):
        self.assertEqual(10, ex.mouse_cortical_7C.size)
        self.assertEqual(['gF', 'gE', 'gP', 'gC', 'gS', 'pF', 'pE',
                          'pP', 'pC', 'pS'],
                         ex.mouse_cortical_7C.names)
        self.assertEqual(ex.mouse_cortical_7C.update([1, 1, 0, 0, 0, 0,
                                                      0, 0, 0, 0], 0),
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(ex.mouse_cortical_7C.update([1, 1, 0, 0, 0, 0,
                                                      0, 0, 0, 0]),
                         [0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(ex.mouse_cortical_7C.update([1, 1, 0, 0, 0, 0,
                                                      0, 0, 0, 0]),
                         [0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(ex.mouse_cortical_7C.update([0, 0, 0, 0, 0, 0,
                                                      0, 1, 1, 1]),
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_mouse_cortical_7C_attractors(self):
        # Assert attractors.
        attractors = (
            # Anterior desired attractor (reachable from desired anterior
            # initial state)
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            # Posterior desired attractor (unreachable from desired posterior
            # initial state)
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        )

        for state in attractors:
            self.assertEqual(ex.mouse_cortical_7C.update(state[:]), state)

        # Assert non-attractors.
        for state in ex.mouse_cortical_7C:
            if state not in attractors:
                self.assertNotEqual(
                    ex.mouse_cortical_7C.update(state[:]), state)

    def test_mouse_cortical_7C_logic_expressions(self):
        self.assertEqual(['gF', 'gE', 'gP', 'gC', 'gS', 'pF', 'pE',
                          'pP', 'pC', 'pS'],
                         ex.mouse_cortical_7C_from_expr.names)
        self.assertEqual(ex.mouse_cortical_7C_from_expr.table,
                         ex.mouse_cortical_7C.table)
        # Assert attractors.
        attractors = (
            # Anterior desired attractor (reachable from desired anterior
            # initial state)
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            # Posterior desired attractor (unreachable from desired
            # posterior initial state)
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        )

        for state in attractors:
            self.assertEqual(
                ex.mouse_cortical_7C_from_expr.update(state[:]), state)

        # Assert non-attractors.
        for state in ex.mouse_cortical_7C:
            if state not in attractors:
                self.assertNotEqual(
                    ex.mouse_cortical_7C_from_expr.update(state[:]), state)
