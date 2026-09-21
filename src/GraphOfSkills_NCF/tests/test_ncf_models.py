import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in dependency-light installs
    torch = None

if torch is not None:
    from gos.ncf import GMF, MLP, NeuMF, NCFConfig, PointwiseNCFLoss, build_ncf_model


@unittest.skipIf(torch is None, "the optional ncf dependency (torch) is not installed")
class NCFModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NCFConfig(
            input_dim=12,
            gmf_latent_dim=4,
            mlp_layer_sizes=(16, 8, 4),
            dropout=0.0,
        )
        generator = torch.Generator().manual_seed(7)
        self.inputs = (
            torch.randn(5, 12, generator=generator),
            torch.randn(5, 12, generator=generator),
        )

    def test_all_models_produce_one_logit_and_probability_per_pair(self) -> None:
        for model_name in ("gmf", "mlp", "neumf"):
            with self.subTest(model=model_name):
                model = build_ncf_model(model_name, self.config)
                logits = model(*self.inputs)
                probabilities = model.predict_proba(*self.inputs)

                self.assertEqual(tuple(logits.shape), (5,))
                self.assertEqual(tuple(probabilities.shape), (5,))
                self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))

    def test_gmf_uses_elementwise_product(self) -> None:
        model = GMF(self.config)
        actual = model.interaction(*self.inputs)
        expected = model.task_projection(self.inputs[0]) * model.skill_projection(
            self.inputs[1]
        )

        self.assertEqual(tuple(actual.shape), (5, self.config.gmf_latent_dim))
        self.assertTrue(torch.equal(actual, expected))

    def test_mlp_uses_halving_relu_tower(self) -> None:
        model = MLP(self.config)
        linear_shapes = [
            (layer.in_features, layer.out_features)
            for layer in model.tower
            if isinstance(layer, torch.nn.Linear)
        ]

        self.assertEqual(model.task_projection.out_features, 8)
        self.assertEqual(model.skill_projection.out_features, 8)
        self.assertEqual(linear_shapes, [(16, 8), (8, 4)])
        self.assertEqual(tuple(model.interaction(*self.inputs).shape), (5, 4))

    def test_neumf_has_independent_gmf_and_mlp_projections(self) -> None:
        model = NeuMF(self.config)

        self.assertIsNot(model.gmf_task_projection, model.mlp_task_projection)
        self.assertIsNot(model.gmf_skill_projection, model.mlp_skill_projection)
        self.assertEqual(
            model.output.in_features,
            self.config.gmf_latent_dim + self.config.mlp_output_dim,
        )

    def test_pretrained_neumf_matches_weighted_gmf_and_mlp_logits(self) -> None:
        gmf = GMF(self.config).eval()
        mlp = MLP(self.config).eval()
        neumf = NeuMF(self.config).eval()
        alpha = 0.35

        neumf.load_pretrained(gmf, mlp, alpha=alpha)

        expected = alpha * gmf(*self.inputs) + (1.0 - alpha) * mlp(*self.inputs)
        self.assertTrue(
            torch.allclose(neumf(*self.inputs), expected, atol=1e-7, rtol=1e-6)
        )

    def test_pointwise_loss_supports_binary_and_graded_targets(self) -> None:
        logits = torch.zeros(4, requires_grad=True)
        grades = torch.tensor([2, 1, 0, -1])
        weights = torch.tensor([1.0, 2.0, 1.0, 100.0])

        binary = PointwiseNCFLoss("binary")(
            logits, grades, sample_weight=weights
        )
        graded = PointwiseNCFLoss("graded")(
            logits, grades, sample_weight=weights
        )

        self.assertTrue(torch.isfinite(binary))
        self.assertTrue(torch.isfinite(graded))
        binary.backward(retain_graph=True)
        graded.backward()
        self.assertIsNotNone(logits.grad)

    def test_invalid_embedding_shape_is_rejected(self) -> None:
        model = NeuMF(self.config)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            model(torch.zeros(2, 12), torch.zeros(3, 12))


if __name__ == "__main__":
    unittest.main()
