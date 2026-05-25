import base64
import json
import zlib
import numpy as np
from scipy.stats import wasserstein_distance
from flwr.server.strategy import FedAvg
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from utils.fipca import ServerFIPCA, reconstruct_delta, flatten_weights, unflatten_weights

class DistanceAwareAggregation(FedAvg):
    def __init__(self, *args, similarity_metric="kl", target_components=50, dp_epsilon=0.0, dp_clip_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_reference_dist = None
        self.similarity_metric = similarity_metric
        self.fipca = ServerFIPCA(target_components=target_components)
        self.global_weights_flat = None
        self.param_shapes = None
        self.current_basis_version = 0
        self.basis_just_changed = False
        self.dp_epsilon = dp_epsilon
        self.dp_clip_norm = dp_clip_norm

    def configure_fit(self, server_round, parameters, client_manager):
        base = super().configure_fit(server_round, parameters, client_manager)
        dp_config = {"dp_epsilon": str(self.dp_epsilon), "dp_clip_norm": str(self.dp_clip_norm)}
        if not self.fipca.is_fitted():
            return [(cp, FitIns(fi.parameters, {"pca_fitted": "0", **dp_config})) for cp, fi in base]
        config = {"pca_fitted": "1", "basis_version": str(self.current_basis_version), **dp_config}
        if self.basis_just_changed:
            b64, n_k_str, d_str = self.fipca.get_basis_b64()
            config["pca_components"] = b64
            config["n_components"]   = n_k_str
            config["original_dim"]   = d_str
        return [(cp, FitIns(fi.parameters, config)) for cp, fi in base]

    def _compute_distances(self, dist_array):
        if self.similarity_metric == "wasserstein":
            return np.array([
                wasserstein_distance(dist_array[i], self.server_reference_dist)
                for i in range(len(dist_array))
            ])
        return np.sum(
            dist_array * np.log((dist_array + 1e-12) / (self.server_reference_dist + 1e-12)),
            axis=1,
        )

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        weights_list = []
        dist_list = []

        for client, fit_res in results:
            compressed = fit_res.metrics.get("compressed", "0") == "1"
            if compressed and self.fipca.is_fitted() and self.global_weights_flat is not None:
                scores = parameters_to_ndarrays(fit_res.parameters)[0]
                delta_flat = reconstruct_delta(scores, self.fipca.pca.components_, self.fipca.pca.mean_)
                full_flat  = self.global_weights_flat + delta_flat
                weights    = unflatten_weights(full_flat, self.param_shapes)
            else:
                weights = parameters_to_ndarrays(fit_res.parameters)
                if self.param_shapes is None and weights:
                    self.param_shapes = [a.shape for a in weights]
                if self.global_weights_flat is None and weights:
                    self.global_weights_flat = flatten_weights(weights)
            weights_list.append(weights)
            if fit_res.metrics and "label_dist" in fit_res.metrics:
                dist_list.append(json.loads(fit_res.metrics["label_dist"]))

        if not dist_list:
            return super().aggregate_fit(server_round, results, failures)

        dist_array = np.array(dist_list)
        if self.server_reference_dist is None:
            self.server_reference_dist = np.mean(dist_array, axis=0)

        distances = self._compute_distances(dist_array)
        inverse_dist = 1.0 / (distances + 1e-6)
        agg_weights = inverse_dist / np.sum(inverse_dist)

        global_weights = [np.average(np.stack(layer), axis=0, weights=agg_weights) for layer in zip(*weights_list)]

        self.server_reference_dist = np.average(dist_array, axis=0, weights=agg_weights)

        new_global_flat = flatten_weights(global_weights)
        prev_k = self.fipca.n_components_current
        self.fipca.update(new_global_flat)
        self.global_weights_flat = new_global_flat
        if self.fipca.is_fitted() and self.fipca.n_components_current != prev_k:
            self.current_basis_version += 1
            self.basis_just_changed = True
        else:
            self.basis_just_changed = self.fipca.basis_changed()

        bytes_before_list = [int(r.metrics.get("bytes_before", "0")) for _, r in results]
        bytes_after_list  = [int(r.metrics.get("bytes_after",  "0")) for _, r in results]
        avg_before = sum(bytes_before_list) / len(bytes_before_list) if bytes_before_list else 0
        avg_after  = sum(bytes_after_list)  / len(bytes_after_list)  if bytes_after_list  else 0
        if avg_before > 0:
            reduction = (1.0 - avg_after / avg_before) * 100.0
            k_str = str(self.fipca.n_components_current) if self.fipca.is_fitted() else "—"
            status = f"K={k_str} | {avg_after/1024:.3f} KB sent vs {avg_before/1024:.1f} KB | Reduction: {reduction:.1f}%"
        else:
            status = "fallback (PCA warming up)"
        print(f"[FIPCA] Round {server_round} | {status}")

        return ndarrays_to_parameters(global_weights), {
            "server_round":   server_round,
            "client_weights": agg_weights.tolist(),
            "metric":         self.similarity_metric,
        }