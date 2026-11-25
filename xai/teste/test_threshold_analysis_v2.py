import unittest

import numpy as np
import pandas as pd

from xai.case_selection import BRANCH_PARA_CENARIO_INDIVIDUAL
from xai.threshold_analysis_v2 import (
    BRANCHES,
    comparar_com_prototipos,
    preparar_predicoes,
    resumir_correlacoes,
    resumir_grupos,
)


class TestLimiarV2(unittest.TestCase):
    def test_contagens_completas_por_corte_rejeitam_predicoes_ausentes_ou_duplicadas(self):
        df = pd.DataFrame([
            (BRANCH_PARA_CENARIO_INDIVIDUAL[branch], seed, sample, sample, 0, 0.52, 0.48)
            for branch in BRANCHES for seed in (7, 12) for sample in (0, 1)
        ], columns=["cenario", "seed", "sample", "y_true", "y_pred", "prob_0", "prob_1"])
        preparado = preparar_predicoes(df, "teste")
        self.assertEqual(len(preparado), 8)
        np.testing.assert_allclose(preparado["margin"], 0.04)
        resumo = resumir_grupos(preparado)
        self.assertEqual(len(resumo), 24)
        self.assertTrue(resumo.groupby(["branch", "margin_cutoff"])["n_observations"].sum().eq(4).all())
        self.assertTrue(resumo.loc[~resumo["near_boundary"], "n_observations"].eq(0).all())
        for invalido in (df.iloc[1:], pd.concat([df, df.iloc[:1]])):
            with self.assertRaises(ValueError):
                preparar_predicoes(invalido, "teste")

    def test_prototipo_exclui_a_propria_imagem_e_o_mesmo_paciente(self):
        grupo = pd.DataFrame({
            "sample": [0, 1, 2, 3, 4, 5],
            "y_pred": [1, 1, 1, 1, 1, 2],
            "correct": [True, True, True, False, True, False],
        })
        mapas = {
            0: np.array([[0, 1], [0, 1]]),
            1: np.array([[0, 0], [1, 1]]),
            2: np.array([[0, 1], [1, 1]]),
            3: np.array([[1, 0], [0, 0]]),
            4: np.zeros((2, 2)),
            5: np.array([[0, 1], [0, 1]]),
        }
        comparados = []

        def ssim(mapa, referencia):
            comparados.append((mapa, referencia))
            return 0.75

        com_paciente, _ = comparar_com_prototipos(
            grupo, mapas, {0: "A", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}, ssim
        )
        self.assertEqual(com_paciente["n_reference_maps"].tolist(), [1, 1, 2, 3, 3, 0])
        np.testing.assert_array_equal(comparados[0][1], mapas[2])
        np.testing.assert_array_equal(comparados[2][1], (mapas[0] + mapas[1]) / 2)
        np.testing.assert_allclose(comparados[3][1], np.mean([mapas[0], mapas[1], mapas[2]], axis=0))
        self.assertEqual(com_paciente["ssim_status"].tolist(), [
            "valido", "valido", "valido", "valido", "mapa_plano", "sem_referencia"
        ])
        self.assertTrue(com_paciente.loc[[4, 5], "ssim"].isna().all())
        self.assertTrue(com_paciente["reference_excludes_patient"].all())

        comparados.clear()
        sem_paciente, _ = comparar_com_prototipos(grupo, mapas, {}, ssim)
        self.assertEqual(sem_paciente.loc[0, "n_reference_maps"], 2)
        np.testing.assert_array_equal(comparados[0][1], (mapas[1] + mapas[2]) / 2)

    def test_spearman_por_seed_conta_cobertura_e_empates(self):
        dados = pd.DataFrame({
            "dataset": "teste",
            "branch": BRANCHES[0],
            "seed": [7] * 4 + [12] * 4,
            "y_pred": 1,
            "margin": [0.1, 0.2, 0.2, 0.9] * 2,
            "ssim": [0.3, 0.4, 0.4, np.nan, 0.2, 0.2, 0.2, 0.2],
        })
        por_seed, resumo = resumir_correlacoes(dados)
        self.assertTrue(por_seed.loc[por_seed["seed"].eq(7), "n_ssim_valid"].eq(3).all())
        self.assertTrue(por_seed.loc[por_seed["seed"].eq(12), "spearman"].isna().all())
        self.assertTrue(resumo["n_seeds_total"].eq(2).all())
        self.assertTrue(resumo["n_seeds_valid"].eq(1).all())
        np.testing.assert_allclose(resumo["spearman_median"], 1)


if __name__ == "__main__":
    unittest.main()
